# this function return img with bounding rectangle using DETR from huggingface

from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import torch
import cv2
import time

model_name = "facebook/detr-resnet-50"

# turn cv2 image to PIL image
def cv2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def getobject(ogimg,threshold=0.9,test=False):
    # get time delta
    start = time.time()

    # get cuda device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgDetect = ogimg.copy()
    print("time delta 1 (copy image): ",time.time()-start)

    # convert input image to PIL image for model input
    pilimg = cv2pil(ogimg)
    print("time delta 2 (cv2pil): ",time.time()-start)

    # create model and feature extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    model.to(device)
    print("time delta 3 (create model and extractor): ",time.time()-start)

    # inferenceogimg
    inputs = feature_extractor(images = pilimg, return_tensors = "pt").to(device)
    outputs = model(**inputs)
    print("time delta 4 (inference): ",time.time()-start)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([pilimg.size[::-1]]).to(device)
    results = feature_extractor.post_process(outputs, target_sizes)[0]
    print("time delta 5 (convert to coco): ",time.time()-start)

    resultsboxes = []

    # draw bounding boxes on the original image
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i) for i in box.tolist()]
        
        if score > threshold:
            resultsboxes.append(box)
            if test:
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box[0]}, {box[1]}, {box[2]}, {box[3]}"
                )

            else:
                cv2.rectangle(imgDetect, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(imgDetect, f"{model.config.id2label[label.item()]}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    print("time delta 6 (draw bounding boxes): ",time.time()-start)
    
    if test:
        return resultsboxes
    else:
        return imgDetect
