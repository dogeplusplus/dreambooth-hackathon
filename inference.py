import numpy as np
import matplotlib.pyplot as plt

from einops import rearrange
from accelerate import Accelerator
from argparse import ArgumentParser
from diffusers import StableDiffusionPipeline


def parse_arguments():
    parser = ArgumentParser("Dreambooth inference script")
    parser.add_argument("--output-dir", type=str, help="Path to model directory")
    parser.add_argument("--prompt", type=str, help="Instance prompt")
    parser.add_argument("--guidance", type=float, help="Guidance scale", default=7)
    parser.add_argument("--images", type=int, help="Number of images to generate", default=1)

    args = parser.parse_args()
    return args


def main(args):
    accelerator = Accelerator()
    pipe = StableDiffusionPipeline.from_pretrained(args.output_dir)
    pipe.to(accelerator.device)
    image_stack = []
    for _ in range(args.images):
        images = pipe(args.prompt, guidance_scale=args.guidance).images[0]
        image_stack.append(np.array(images))

    image_stack = np.stack(image_stack)
    image_grid = rearrange(image_stack, "(y x) h w c -> (y h) (x w) c", x=min(4, args.images))
    plt.imshow(image_grid)
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)