# this function return img with bounding rectangle using DETR from huggingface

from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
import cv2

model_name = "facebook/detr-resnet-50"

def getobject(ogimg,threshold):
    imgDetect = ogimg.copy()
    # create model and feature extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)

    # inferenceogimg
    inputs = feature_extractor(images = ogimg, return_tensors = "pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([ogimg.shape[::-1]])
    results = feature_extractor.post_process(outputs, target_sizes)

    # draw bounding boxes on the original image
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i,2) for i in box.tolist()]
    
        if score > threshold:
            cv2.rectangle(imgDetect, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(imgDetect, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return imgDetect
