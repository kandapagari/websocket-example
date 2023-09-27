# -*- coding: utf-8 -*-
import requests
import torch
from PIL import Image
from transformers import (AutoImageProcessor,
                          Mask2FormerForUniversalSegmentation)


class Mask2Former(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-tiny-coco-instance")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-tiny-coco-instance")

    @property
    def labels(self):
        return open("assets/coco-labels-2014_2017.txt").read().splitlines()

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        return self.processor.post_process_instance_segmentation(
            outputs,
            overlap_mask_area_threshold=0.9,
            mask_threshold=0.9,
            threshold=0.95,
            target_sizes=[image.size[::-1]],
        )[0]


if __name__ == "__main__":
    mask2former = Mask2Former()
    url = "http://images.cocodataset.org/test2017/000000000674.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    instance_map = mask2former.forward(image)
    print(instance_map)
