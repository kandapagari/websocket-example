# -*- coding: utf-8 -*-
import requests
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


class Detr(torch.nn.Module):

    def __init__(self,
                 *args,
                 model_name="facebook/detr-resnet-50",
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9)[0]

        for score, label, box in zip(results["scores"], results["labels"],
                                     results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}")


if __name__ == "__main__":
    detr = Detr()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    detr.forward(image)
    detr = Detr(model_name="facebook/detr-resnet-101")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    detr.forward(image)
