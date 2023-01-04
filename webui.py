import numpy as np
import gradio as gr

from torch import autocast
from einops import rearrange
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline


accelerator = Accelerator()
pipe = StableDiffusionPipeline.from_pretrained("jaffa-cake")
pipe.safety_checker = lambda images, **kwargs: (images, False)
pipe.to(accelerator.device)


def tile_images(images):
    h, w, c = images[0].shape
    dim = int(np.sqrt(len(images)))

    image_grid = np.zeros((dim, dim, h, w, c), dtype=np.uint8)
    for i in range(dim):
        for j in range(dim):
            x = i * dim + j
            image_grid[i, j] = images[x]

    image_grid = rearrange(image_grid, "y x h w c -> (y h) (x w) c", x=dim)
    return image_grid


def inference(prompt, guidance, num_images, seed):
    set_seed(int(seed))
    image_stack = []
    with autocast("cuda"):
        while len(image_stack) < num_images:
            image = pipe(prompt, guidance_scale=guidance).images[0]
            image_stack.append(np.array(image))

    image_grid = tile_images(image_stack)

    return [image_grid] + image_stack


app = gr.Interface(
    fn=inference,
    inputs=[
        "text",
        gr.Slider(0, 100, value=7),
        gr.Slider(1, 32, step=1),
        gr.Number(value=42),
    ],
    outputs=gr.Gallery()
)
app.launch()