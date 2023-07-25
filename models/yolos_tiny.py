# -*- coding: utf-8 -*-
import requests
import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor


class YoloTiny(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = YolosForObjectDetection.from_pretrained(
            'hustvl/yolos-tiny')
        self.image_processor = YolosImageProcessor.from_pretrained(
            "hustvl/yolos-tiny")

    def forward(self, image):

        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # print results
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=0.9, target_sizes=target_sizes)[0]
        for score, label, box in zip(results["scores"], results["labels"],
                                     results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}")


if __name__ == "__main__":
    yolos = YoloTiny()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    yolos.forward(image)
