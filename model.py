from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
from torch import nn


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
        depth = Image.fromarray(formatted)
        return depth

    def dry_run(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        self.forward(image)
        return "Dry run complete"
