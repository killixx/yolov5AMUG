
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import scoreboard 
import regex 

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'model1.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
   # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs


    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # List of input frames
input_frames = ['frame1.jpg', 'frame2.jpg', 'frame3.jpg']

class Process:
    def __init__(self,save_conf,names,windows,hide_conf,hide_labels,im0s,dataset,line_thickness,model,dt,save_dir,path,augment,conf_thres,iou_thres,classes, agnostic_nms,max_det,save_crop,save_txt,save_img,view_img, img_size, im0, results, agnostic, colors, img, device):
        self.save_conf = save_conf
        self.names = names
        self.windows = windows
        self.hide_conf = hide_conf
        self.hide_labels = hide_labels
        self.im0s = im0s
        self.dataset = dataset
        self.line_thickness = line_thickness
        self.model = model
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.path = path
        self.save_dir = save_dir
        self.dt = dt
        self.img_size = img_size
        self.im0 = im0
        self.results = results
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic = agnostic
        self.classes = classes
        self.colors = colors
        self.save_img = save_img
        self.save_crop = save_crop
        self.view_img = view_img
        self.save_txt = save_txt
        self.img = img
        self.device = device

    def initialize(self):
        self.width = 1280
        self.height = 720
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.img_size = 640
        self.device = ''
        self.view_img = False
        self.save_txt = False
        self.classes = []
        self.colors = [[0, 255, 0]]
        self.aggregation = 'first'
        #self.dt =
        #self.model = 

process1=Process()
process1.initialize()

process2=Process()
process2.initialize()

# Loop through input frames
for frame in input_frames:
    # Call frame_processing function to detect and extract scoreboard
    output_im, ratio, (xyxy, labels, scores) = scoreboard.frame_processing(model,dt,save_dir,path,augment,conf_thres,iou_thres,classes, agnostic_nms,max_det,webcam,im0s,dataset,line_thickness,save_crop,hide_conf,hide_labels,save_txt,save_img,save_conf,view_img,names,windows)

    # Call regex_processing function to extract information from scoreboard
    scoreboard_info = regex.score_processing(output_im, xyxy)

    # Save result
    with open(f'{frame}_info.txt', 'w') as f:
        f.write(scoreboard_info)

    