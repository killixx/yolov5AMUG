
import argparse
import os
import platform
import sys
from pathlib import Path
from PIL import Image
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
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, LoadCroppedImage
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box, get_cropped_box
from utils.torch_utils import select_device, smart_inference_mode

class ImageProcess:
    
    #region Variables
    
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    windows=[]
    frame = None
    model1 = None
    model2 = None
    max_det = 1#maximum detections per image.e.g.1000
    agnostic_nms = False
    augment = False
    dt = (Profile(), Profile(), Profile())
    imgsz1 = (640, 640)
    imgsz2 = (640, 640)
    conf_thres = 0.4
    iou_thres = 0.5
    classes = None
    half=False  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
    vid_stride=1
    stride1 = None
    stride2 = None
    names1 = None
    names2 = None
    device = ''
    bs = 1 # Batch Size
    pt1 = None
    pt2 = None
    #endregion
    
    #region Methods
    
    def initialize(self, weights_one_path, weights_two_path):
        self.device = select_device(self.device)
        
        #initializing model
        self.model1 = DetectMultiBackend(weights= weights_one_path, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.model2 = DetectMultiBackend(weights= weights_two_path, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        
        #initializin strides and names and pt
        self.stride1, self.names1, self.pt1 = self.model1.stride, self.model1.names, self.model1.pt
        self.stride2, self.names2, self.pt2 = self.model2.stride, self.model2.names, self.model2.pt
        
        #imageSize
        self.imgsz1 = check_img_size(self.imgsz1, s=self.stride1)  # check image size
        self.imgsz2 = check_img_size(self.imgsz2, s=self.stride2)  # check image size
        
        # Run inference
        self.model1.warmup(imgsz=(1 if self.pt1 or self.model1.triton else self.bs, 3, *self.imgsz1))  # warmup
        self.model2.warmup(imgsz=(1 if self.pt2 or self.model2.triton else self.bs, 3, *self.imgsz2))  # warmup
        
        
    def GetImageSizes(self):
        return self.imgsz1, self.imgsz2
        
    def GetPt(self):
        return self.pt1, self.pt2
        
    def GetStrides(self):
        return self.stride1, self.stride2
        
    def DetectImage(self,im, im0s, is_detecting_board = False):
        
        results = []
        
        with self.dt[0]:
            if is_detecting_board:
                im = torch.from_numpy(im).to(self.model1.device)
                im = im.half() if self.model1.fp16 else im.float()  # uint8 to fp16/32
            else:
                im = torch.from_numpy(im).to(self.model2.device)
                im = im.half() if self.model2.fp16 else im.float()  # uint8 to fp16/32
                
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            pred = self.model1(im, augment=self.augment, visualize=False)

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.ou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)


        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = im0s.copy()
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()
            annotator = Annotator(im0, line_width=3,example="")
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None
                    annotator.box_label(xyxy,"", color=colors(c, True))
                    crop=get_cropped_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True , save = False)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    kernel = np.ones((3,3),np.uint8)
                    erosion = cv2.erode(binary,kernel,iterations = 1)
                    denoised = cv2.fastNlMeansDenoising(erosion, None, h=10, templateWindowSize=7, searchWindowSize=21)
                    
                    results.append(denoised)
                    
        return results

        
    #endregion



####Program Execution which needs to be moved to a separate file

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights1', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--weights2', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)



@smart_inference_mode()
def run(
        weights1=ROOT / 'yolov5s.pt',  # model path or triton URL
        weights2=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    imageProcessor = ImageProcess()
    imageProcessor.initialize(weights1, weights2)
    imgsz1, imgsz2 = imageProcessor.GetImageSizes()
    pt1, pt2 = imageProcessor.GetPt()
    stride1, stride2 = imageProcessor.GetStrides()

    if screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz1, stride=stride1, auto=pt1)
    else:
        dataset = LoadImages(source, img_size=imgsz1, stride=stride1, auto=pt1, vid_stride=1)

    for path, im, im0, vid_cap, s in dataset:
        result = imageProcessor.DetectImage(im, im0, True)
        
        f = str(".image".with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(result[0][..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
        
        if len(result) > 0:
            for frame in result:
                frame0 = LoadCroppedImage(f, imgsz2, stride2, pt2)
                path, im, im0, vid_cap, s = frame0.get_results()
                
                #detecting Scores
                resuts = imageProcessor.DetectImage(im,im0, False)
                
                #TODO: Further processing for converting image to text and Keeping tracks of score

                
        



    