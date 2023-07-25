# -*- coding: utf-8 -*-
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class GIT(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/git-base-coco")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/git-base-coco")

    def forward(self, image):
        pixel_values = self.processor(images=image,
                                      return_tensors="pt").pixel_values

        generated_ids = self.model.generate(pixel_values=pixel_values,
                                            max_length=50)
        generated_caption = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return generated_caption

    def dry_run(self):
        url = "http://images.cocodataset.org/test2017/000000581864.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        print(self.forward(image))
        return "Dry run complete"


if __name__ == "__main__":
    git_model = GIT()
    print(git_model.dry_run())
