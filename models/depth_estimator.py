# -*- coding: utf-8 -*-
import numpy as np
import requests
import torch
from PIL import Image
from torch import nn
from transformers import DPTForDepthEstimation, DPTImageProcessor


class DepthEstimator(nn.Module):

    def __init__(self):
        super().__init__()
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    def forward(self, image):
        # prepare image for the model
        inputs = self.processor(images=image, return_tensors="pt")  # type: ignore
        with torch.no_grad():
            outputs = self.model(**inputs)  # type: ignore
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        return Image.fromarray(formatted)

    def dry_run(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        self.forward(image)
        return "Dry run complete"
