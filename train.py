import math
import torch
import bitsandbytes as bnb
import torch.nn.functional as F

from tqdm import trange
from PIL import Image
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from argparse import ArgumentParser
from accelerate.utils import set_seed
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


class DreamBoothDataset(Dataset):
    def __init__(self, dataset, instance_prompt, tokenizer, size=512):
        self.dataset = dataset
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size

        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        image = self.dataset[index]
        example["instance_images"] = self.transforms(image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example


def parse_arguments():
    parser = ArgumentParser("Dreambooth training script")
    parser.add_argument("--files", type=str, help="Path to training images")
    parser.add_argument("--output-dir", type=str, help="Path to save dreambooth models.")
    parser.add_argument("--instance-prompt", type=str, help="Instance prompt to use")
    parser.add_argument("--lr", type=float, help="Learning rate", default=2e-6)
    parser.add_argument(
        "--steps", type=int, help="Number of training steps", default=500
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        help="Number of gradient accumulation steps",
        default=1,
    )
    parser.add_argument("--seed", type=int, help="RNG seed", default=42)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=1)
    parser.add_argument(
        "--max-grad-norm", type=float, help="Gradient clip bound", default=1.0
    )
    parser.add_argument(
        "--gradient-checkpoint",
        action="store_true",
        help="Apply gradient checkpointing",
        default=False,
    )
    parser.add_argument(
        "--use-8bit-adam",
        action="store_true",
        help="Use 8-bit adam optimizer",
        default=False,
    )
    args = parser.parse_args()
    return args


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation,
    )

    model_id = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    set_seed(args.seed)

    if args.gradient_checkpoint:
        unet.enable_gradient_checkpointing()

    if args.use_8bit_adam:
        optimizer = optim.AdamW(unet.parameters(), lr=args.lr)
    else:
        optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr = args.lr)

    noise_scheduler = DDPMScheduler(
        beta_start = 0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    files = list(Path(args.files).rglob("*.png"))
    images = [Image.open(str(p)).convert(mode="RGB") for p in files]
    dataset = DreamBoothDataset(images, args.instance_prompt, tokenizer)

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        return batch

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)

    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    update_steps_per_epoch = math.ceil(len(train_loader) / args.accumulation)
    train_epochs = math.ceil(args.steps / update_steps_per_epoch)

    VAE_LATENT_FACTOR = 0.18215

    pbar = trange(train_epochs, disable=not accelerator.is_local_main_process)
    global_step = 0
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        unet.train()
        for batch in train_loader:
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents *= VAE_LATENT_FACTOR
                
                noise = torch.randn(latents.shape).to(latents.device)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (batch_size,),
                    device = latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    global_step += 1

                logs = {"loss": loss.detach().item(), "step": global_step}
                pbar.set_postfix(**logs)
    
                optimizer.step()
                optimizer.zero_grad()

            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        scheduler = PNDMScheduler(
            beta_start=8.5e-4,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            steps_offset=1,
        )

        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)