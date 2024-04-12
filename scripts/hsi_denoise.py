import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import yaml
import cv2 as cv
from functools import partial

import sys
sys.path.append("../")
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_hsi_data
from torchvision import utils
from measurement import measurement_fn
from utils import calc_psnr, calc_ssim


def estimate_sigma(image, k, k_m=1):
    assert k > k_m
    image = np.array(image.to("cpu"))
    image = np.transpose(image, [1, 2, 0])
    blur = cv.blur(image, (k, k))
    v = np.var(blur - image, axis=(0, 1))
    std = np.sqrt(v*(k**2)/(k**2-k_m**2))
    return th.from_numpy(std).to(dist_util.dev())[..., None, None]

def median_blur(image, k):
    image = np.array(image.to("cpu"))
    image = np.transpose(image, [1, 2, 0])
    blur = cv.medianBlur(image, k)
    blur = np.transpose(blur, [2, 0, 1])
    return th.from_numpy(blur).to(dist_util.dev())[None]


# added
def load_noise_hsi(data_dir, batch_size):
    data = load_hsi_data(
        data_dir=data_dir,
        batch_size=batch_size,
        deterministic=True,
        mode="test",
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch[1][0]
        yield model_kwargs, large_batch[0][0]


def main():
    args = create_argparser().parse_args()
    if args.model_config is not None:
        upgrade_by_config(args)
    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    logger.log(args)

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")

    logger.log("loading data...")
    data = load_noise_hsi(
        args.base_samples,
        args.batch_size,
    )

    logger.log("creating samples...")
    count = 0
    while count * args.batch_size < args.num_samples:
        model_kwargs, target = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sigma = estimate_sigma(model_kwargs["ref_img"], args.k)
        
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, args.in_channels, *target.shape[-2:]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            measure_fn=partial(measurement_fn, var=sigma**2),
            range_t=args.range_t,
            progress=True,
        )

        sample = (sample + 1)/2
        out_path = os.path.join(logger.get_dir(),
                                f"{str(count).zfill(5)}.png")
        utils.save_image(
            sample[0, 0].unsqueeze(0),
            out_path,
            nrow=1,
            normalize=True,
            range=(0, 1),
        )

        out_path = os.path.join(logger.get_dir(),
                                f"{str(count).zfill(5)}.npy")
        np.save(out_path, sample.cpu().numpy())

        target = (target + 1)/2
        out_path = os.path.join(logger.get_dir(),
                                f"target-{str(count).zfill(5)}.png")
        utils.save_image(
            target[0].unsqueeze(0),
            out_path,
            nrow=1,
            normalize=True,
            range=(0, 1),
        )

        noise = model_kwargs['ref_img']
        noise = (noise + 1)/2
        out_path = os.path.join(logger.get_dir(),
                                f"noise-{str(count).zfill(5)}.png")
        utils.save_image(
            noise[0].unsqueeze(0),
            out_path,
            nrow=1,
            normalize=True,
            range=(0, 1),
        )
        logger.log("PSNR:")
        logger.log(f"count:{count}, before: {calc_psnr(target.numpy(), noise.cpu().numpy())}")
        logger.log(f"count:{count}, after: {calc_psnr(target.numpy(), sample.cpu().numpy()[0])}")
        logger.log(f"count:{count}, before: {calc_ssim(target.numpy(), noise.cpu().numpy())}")
        logger.log(f"count:{count}, after: {calc_ssim(target.numpy(), sample.cpu().numpy()[0])}")
        count += 1
        logger.log(f"created {count * args.batch_size} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=5,
        batch_size=1,
        range_t=0,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_dir="",
        model_config=None,
        k=5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def upgrade_by_config(args):
    model_config = load_yaml(args.model_config)
    for k, v in model_config.items():
        setattr(args, k, v)


if __name__ == "__main__":
    main()