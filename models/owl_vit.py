# -*- coding: utf-8 -*-
import requests
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


class OwlVit(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )

    def forward(self, image, texts, *, threshold=0.5):
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )
        return results


if __name__ == "__main__":
    model = OwlVit()
    url = "http://images.cocodataset.org/test2017/000000000674.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = [["person", "red dress"]]
    results = model.forward(image, texts, threshold=0.25)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = (
        results[i]["boxes"],
        results[i]["scores"],
        results[i]["labels"],
    )
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
        )
