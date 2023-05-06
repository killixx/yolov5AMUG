import os
import argparse
import torch
import torchvision
from pathlib import Path
from google.colab import drive
from PIL import Image

# Mount Google Drive
drive.mount('/content/gdrive')

def detect(save_img=False):
    # Set up model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.4  # confidence threshold (default 0.25)
    model.iou = 0.45  # NMS IoU threshold (default 0.45)

    # Set up folder paths
    source_folder = '/content/gdrive/MyDrive/Images'
    output_folder = '/content/gdrive/MyDrive/Detected_Objects'

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through images in source folder
    for img_path in Path(source_folder).glob('**/*.jpg'):
        # Load image
        img = Image.open(str(img_path))

        # Perform detection
        results = model(img, size=640)

        # Save image with bounding boxes if save_img is True
        if save_img:
            save_path = str(img_path).replace(source_folder, output_folder)
            results.render(save_path, render=False, save_img=True)

        # Save detected objects as new images
        for *xyxy, conf, cls in reversed(results.xyxy[0]):
            x1, y1, x2, y2 = [int(x) for x in xyxy]
            cropped_img = img.crop((x1, y1, x2, y2))
            save_path = str(img_path).replace(source_folder, output_folder).replace('.jpg', f'_{cls}.jpg')
            cropped_img.save(save_path)

if __name__ == '__main__':
    detect(save_img=True)