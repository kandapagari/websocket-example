# -*- coding: utf-8 -*-
from enum import Enum

import requests
import torch
from PIL import Image
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor


class TaskInput(Enum):
    semantic = "semantic"
    instance = "instance"
    panoptic = "panoptic"


class OneFormer(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_coco_swin_large")
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_coco_swin_large")
        self.func_map = dict(
            semantic=self.processor.post_process_semantic_segmentation,
            instance=self.processor.post_process_instance_segmentation,
            panoptic=self.processor.post_process_panoptic_segmentation)

    @property
    def labels(self):
        return open("assets/coco-labels-2014_2017.txt").read().splitlines()

    def forward(self, image, task: TaskInput = TaskInput.instance):
        inputs = self.processor(images=image,
                                task_inputs=[task.value],
                                return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        return self.func_map[task.value](outputs, target_sizes=[image.size[::-1]])[
            0
        ]


if __name__ == "__main__":
    oneformer = OneFormer()
    url = "http://images.cocodataset.org/test2017/000000000674.jpg"
    # url = 'https://huggingface.co/datasets/shi-labs/oneformer_demo/blob/main/coco.jpeg'
    image = Image.open(requests.get(url, stream=True).raw)
    instance_map = oneformer.forward(image, task=TaskInput.instance)
    semantic_map = oneformer.forward(image, task=TaskInput.semantic)
    panoptic_map = oneformer.forward(image, task=TaskInput.panoptic)
    print(semantic_map)
    print(instance_map)
    print(panoptic_map)
