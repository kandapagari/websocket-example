# -*- coding: utf-8 -*-
import torch
from controlnet_aux.canny import CannyDetector
from diffusers import (AutoencoderKL, EulerAncestralDiscreteScheduler,
                       StableDiffusionXLAdapterPipeline, T2IAdapter)
from diffusers.utils import load_image


class T2iAdapterCanny(torch.nn.Module):
    def __init__(self, *args, device='cpu', **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # load adapter
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            varient="fp16"
        ).to(device)

        # load euler_a scheduler
        model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id,
            vae=vae,
            adapter=adapter,
            scheduler=euler_a,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)
        self.pipe.enable_xformers_memory_efficient_attention()

        self.canny_detector = CannyDetector()

    def forward(self, image, prompt, neg_prompt):

        # Detect the canny map in low resolution to avoid high-frequency details
        image = self.canny_detector(image, detect_resolution=384,
                                    image_resolution=1024)  # .resize((1024, 1024))

        gen_images = self.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=image,
            num_inference_steps=30,
            guidance_scale=7.5,
            adapter_conditioning_scale=0.8,
            adapter_conditioning_factor=1
        )
        return gen_images.images


if __name__ == "__main__":
    url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_canny.jpg"
    image = load_image(url)
    t2i_adapter_model = T2iAdapterCanny(device='cuda')
    prompt = "Mystical fairy in real, magic, 4k picture, high quality"
    neg_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch,\
        deformed, mutated, ugly, disfigured"
    gen_images = t2i_adapter_model(image, prompt, neg_prompt)
    gen_images[0].save('out_canny.png')
