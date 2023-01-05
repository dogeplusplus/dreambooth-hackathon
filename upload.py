from huggingface_hub import HfApi, ModelCard, create_repo, get_full_repo_name

model_name = "jaffa-cake"
hub_model_id = get_full_repo_name(model_name)
create_repo(hub_model_id)
api = HfApi()
api.upload_folder(folder_path="jaffa-cake", path_in_repo="", repo_id=hub_model_id)
api.upload_folder(folder_path="assets", path_in_repo="assets", repo_id=hub_model_id)

prompt = "a photo of a jaffa cake"
name_of_your_concept = "jaffa"
theme = "food"
description = "Generate images of jaffa cakes in different contexts and art styles:"
instance_prompt = "a photo of a jaffa cake"
content = f"""
---
license: creativeml-openrail-m
tags:
- pytorch
- diffusers
- stable-diffusion
- text-to-image
- diffusion-models-class
- dreambooth-hackathon
- {theme}
widget:
- text: {prompt}
---

# DreamBooth model for the {name_of_your_concept} concept trained by {api.whoami()["name"]} on some images of jaffa cakes.

This is a Stable Diffusion model fine-tuned on the {name_of_your_concept} concept with DreamBooth. It can be used by modifying the `instance_prompt`: **{instance_prompt}**

This model was created as part of the DreamBooth Hackathon 🔥. Visit the [organisation page](https://huggingface.co/dreambooth-hackathon) for instructions on how to take part!

## Description

| Stable Diffusion v1-4 | Dreambooth Jaffa |
| -- | -- |
|  ![](stable-jaffa.png) | ![](dreambooth-jaffa.png) |

{description}

| ukiyo-e | picasso | rembrandt |
| -- | -- | -- |
| ![](ukiyoe-jaffa.png) | ![](picasso-jaffa.png) | ![](rembrandt-jaffa.png) |
| **van gogh** | **warhol** | **dali** |
| ![](vangogh-jaffa.png) | ![](warhol-jaffa.png) | ![](dali-jaffa.png) |

## Usage

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained('{hub_model_id}')
image = pipeline().images[0]
image
```
"""

card = ModelCard(content)
hub_url = card.push_to_hub(hub_model_id)