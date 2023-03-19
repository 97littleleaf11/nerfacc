"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from radiance_fields.mlp import VanillaNeRFRadianceField
from utils import render_image, set_random_seed

from nerfacc import ContractionType, OccupancyGrid

device = "cuda:0"
set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="trainval",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="the data path of the pretrained model",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=[
        # nerf synthetic
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
        # mipnerf360 unbounded
        "garden",
    ],
    help="which scene to use",
)
parser.add_argument(
    "--dataset_type",
    type=str,
    default="blender",
    choices=["blender", "llff", "360"],
    help="which kind of dataset to use",
)
parser.add_argument(
    "--aabb",
    type=lambda s: [float(item) for item in s.split(",")],
    default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
    help="delimited list input",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=8192,
)
parser.add_argument(
    "--unbounded",
    action="store_true",
    help="whether to use unbounded rendering",
)
parser.add_argument(
    "--auto_aabb",
    action="store_true",
    help="whether to automatically compute the aabb",
)
parser.add_argument("--cone_angle", type=float, default=0.0)
args = parser.parse_args()

render_n_samples = 1024

# setup the dataset
train_dataset_kwargs = {}
test_dataset_kwargs = {}

if args.dataset_type == "llff":
    train_dataset_kwargs["ndc"] = True
    test_dataset_kwargs["ndc"] = True

if args.dataset_type == "llff" or args.dataset_type == "360":
    from datasets.nerf_360_v2 import SubjectLoader

    target_sample_batch_size = 1 << 16
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    grid_resolution = 128
else:
    from datasets.nerf_synthetic import SubjectLoader

    target_sample_batch_size = 1 << 16
    grid_resolution = 128

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=target_sample_batch_size // render_n_samples,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    **test_dataset_kwargs,
)

if args.auto_aabb:
    if train_dataset_kwargs["ndc"]:
        train_aabb = train_dataset.cal_ndc_aabb()
        test_aabb = test_dataset.cal_ndc_aabb()
        args.aabb = [min(x, y) for x, y in zip(train_aabb[:3], test_aabb[:3])] + [max(x, y) for x, y in zip(train_aabb[3:], test_aabb[3:])]
    else:
        camera_locs = torch.cat(
            [train_dataset.camtoworlds, test_dataset.camtoworlds]
        )[:, :3, -1]
        args.aabb = torch.cat(
            [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
        ).tolist()
    print("Using auto aabb", args.aabb)

# setup the scene bounding box.
if args.dataset_type == "llff":
    print("Using faceforwarding rendering")
    contraction_type = ContractionType.AABB
    scene_aabb = None
    near_plane = 0.01
    far_plane = 1
    render_step_size = 2 / render_n_samples
elif args.unbounded:
    print("Using unbounded rendering")
    contraction_type = ContractionType.UN_BOUNDED_SPHERE
    scene_aabb = None
    near_plane = 0.2
    far_plane = 1e4
    render_step_size = 1e-2
else:
    contraction_type = ContractionType.AABB
    scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
    near_plane = None
    far_plane = None
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / render_n_samples
    ).item()

# setup the radiance field we want to train.
max_steps = 50000
grad_scaler = torch.cuda.amp.GradScaler(1)
radiance_field = VanillaNeRFRadianceField().to(device)
optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[
        max_steps // 2,
        max_steps * 3 // 4,
        max_steps * 5 // 6,
        max_steps * 9 // 10,
    ],
    gamma=0.33,
)

occupancy_grid = OccupancyGrid(
    roi_aabb=args.aabb,
    resolution=grid_resolution,
    contraction_type=contraction_type,
).to(device)

if args.model_path is not None:
    checkpoint = torch.load(args.model_path)
    radiance_field.load_state_dict(checkpoint['radiance_field_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    occupancy_grid.load_state_dict(checkpoint['occupancy_grid_state_dict'])
    step = checkpoint['step']
else:
    step = 0

# training
tic = time.time()
for epoch in range(10000000):
    for i in range(len(train_dataset)):
        radiance_field.train()
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        # update occupancy grid
        occupancy_grid.every_n_step(
            step=step,
            occ_eval_fn=lambda x: radiance_field.query_opacity(
                x, render_step_size
            ),
        )

        # render
        rgb, acc, depth, n_rendering_samples = render_image(
            radiance_field,
            occupancy_grid,
            rays,
            scene_aabb,
            # rendering options
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=args.cone_angle,
        )
        if n_rendering_samples == 0:
            continue

        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)
        alive_ray_mask = acc.squeeze(-1) > 0

        # compute loss
        loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        if step % 5000 == 0:
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.5f} | "
                f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |"
            )

        if step > 0 and step % max_steps == 0:
            # evaluation
            radiance_field.eval()

            psnrs = []
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(test_dataset))):
                    data = test_dataset[i]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]

                    # rendering
                    rgb, acc, depth, _ = render_image(
                        radiance_field,
                        occupancy_grid,
                        rays,
                        scene_aabb,
                        # rendering options
                        near_plane=None,
                        far_plane=None,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=args.cone_angle,
                        # test options
                        test_chunk_size=args.test_chunk_size,
                    )
                    mse = F.mse_loss(rgb, pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                    # imageio.imwrite(
                    #     "acc_binary_test.png",
                    #     ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                    # )
                    imageio.imwrite(
                        f"rgb_test_{step}_{i}.png",
                        (rgb.cpu().numpy() * 255).astype(np.uint8),
                    )
                    # break
            psnr_avg = sum(psnrs) / len(psnrs)
            print(f"evaluation: psnr_avg={psnr_avg}")
            train_dataset.training = True

        if step == max_steps:
            print("training stops")
            exit()

        step += 1
