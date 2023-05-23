
import argparse
import os
import platform
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import scoreboard 
import regex 
from pytesseract import pytesseract
import re
import cv2
import subprocess
import ffmpeg



import torch

pytesseract.tesseract_cmd = 'C:\\Tesseract-OCR\\tesseract.exe'


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
            if is_detecting_board:
                pred = self.model1(im, augment=self.augment, visualize=False)
            else:
                pred = self.model2(im, augment=self.augment, visualize=False)

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)


        # Process predictions
        for i, det in enumerate(pred):  # per image
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
                    crop=get_cropped_box(xyxy,imc,BGR=True)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    if not is_detecting_board:
                        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        kernel = np.ones((3,3),np.uint8)
                        erosion = cv2.erode(binary,kernel,iterations = 1)
                        denoised = cv2.fastNlMeansDenoising(erosion, None, h=10, templateWindowSize=7, searchWindowSize=21)
                        
                        results.append(denoised)
                    else:
                        results.append(crop)
        return results

        
    #endregion

runs = 0
wickets = 0

def readandreturn(results):
    global runs, wickets
    # Apply pytesseract OCR
    text = pytesseract.image_to_string(results[0],config ='--psm 6')

    text.strip()
    text.replace(" ", "")

    pattern = "([0 - 9] * -[0 - 9] *)"

    results = re.match(pattern, text)

    if results == False:
        return runs, wickets
    print(f"Text: {text} ")
    if text != None:
        print(f"Textdetected: {text} ")
        text = text.split('-')
        if len(text) != 2:
            return runs, wickets
        if(text.split('-')[0].strip().isnumeric() == False):
            return runs, wickets
        runs1 = text.split('-')[0].strip()
        if(text.split('-')[1].strip().isnumeric() == False):
            return runs, wickets
        wickets1 = text.split('-')[1].strip()
        runs2=int(runs1)
        wickets2=int(wickets1)
        print(f"runs{runs2}--wickets{wickets2}")
        return runs2,wickets2

def change_detect(detected_runs, detected_wickets):
    global runs, wickets, old_runs, old_wickets

    old_runs = runs
    old_wickets = wickets

    runs = detected_runs
    wickets = detected_wickets

    runs_diff = runs-old_runs
    wickets_diff = wickets-old_wickets

    if(runs_diff > 3 and runs_diff < 7):
        return 

    if(wickets_diff > 0 and wickets_diff < 2):
        print("wicket down")

    

video_path = "video.mp4"
# Create a copy of the source video file
copy = video_path + ".copy"

print (copy)

Y = video_path
        

def run(
        weights1=ROOT / 'model1.pt',  # model path or triton URL
        weights2=ROOT / 'model2.pt',  # model path or triton URL
        source = Y ,  # file/dir/URL/glob/screen/0(webcam)
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
    
    count = 0
    frame = []

    for path, im, im0, vid_cap, s in dataset:
        result = imageProcessor.DetectImage(im, im0, True)
        
        f = ".croppedimage.jpg"
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        
        if len(result) > 0:
            for frame in result:
                cv2.imwrite( f ,  frame)
                frame0 = LoadCroppedImage(f, imgsz2, stride2, pt2)
                path, im, im0, vid_cap, s = frame0.get_results()

                #detecting Scores
                results = imageProcessor.DetectImage(im,im0, False)
                if(len(results))>0:
                    #results wali if k andr ye type kro
                    cv2.imwrite("output.jpg",results[0])
                    cv2.imshow("output.jpg",results[0])
                    d_runs, d_wic = readandreturn(results[0])
                    change_detect(d_runs,d_wic)
        count = count + 1
    # Create a new folder for the short clips
    if not os.path.exists("video_shot"):
        os.mkdir("video_shot")

    # Create a list to store the short clips
    short_clips = []
    frame = frame + (count,)
    i = 0
    for i  in frame[i]:
        starting_duration = 0
        ending_duration = 0
        s = i-210
        e = i+210
        l=len(source)
        # Get the starting and ending duration of the short clip
        if s is True:
            starting_duration = 0
        else:
            starting_duration = i - 210

        if any(e > l):
            ending_duration = l
        else:
            ending_duration = i + 210
        # Create the command to merge the short clip
        cmd = f"ffmpeg -i {source} -ss {str(starting_duration)} -to {str(ending_duration)} -c:v copy -c:a copy shot{i}.mp4"
        print(cmd)
        os.system(f'cmd /c "{cmd}"')
        # Save the short clip in the video_shot folder
        os.system("mv shot" + str(i) + ".mp4 video_shot")

        # Add the short clip to the list
        short_clips.append("shot" + str(i) + ".mp4")
        
        # Display it on the web video display panel
        #print("The video shot for wicket " + str(i) + " is now available on the web video display panel.")


    # Merge the short clips into a single video
    cmd = f"ffmpeg -f concat -i {'|'.join(short_clips)} -c copy highlight.mp4"

    # Display the merged video on the web video display panel
    print("The highlight video is now available on the web video display panel.") 


    print('execution complete')


####Program Execution which needs to be moved to a separate file

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights1', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--weights2', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    opt = parser.parse_args()
    #opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)



def capture_video(video_path):
    video_path = input("Enter the path to the video file: ")
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    # Check if the video capture is successful
    if not cap.isOpened():
        print("Error opening video file.")
        exit()

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)            

def detect_merge(text,total_frames):
    text.strip()
    text.replace(" ", "")

    pattern = "([0 - 9] * -[0 - 9] *)"

    result = re.match(pattern, text)

    if result == False:
        return
    print("Text: "+text)
    if text == None:
        return

    runs = 0
    wickets = 0
    changes = []

    for line in text.split("\n"):
        if "-" in line:
            left, right = line.split("-")
            try:
                runs_diff = int(left.strip())
                wickets_diff = int(right.strip())

                if runs_diff in [4, 6]:
                    runs += runs_diff
                    changes.append(f"Runs changed by {runs_diff} (+{runs_diff})")

                if wickets_diff in [-1, 1]:
                    wickets += wickets_diff
                    changes.append(f"Wickets changed by {wickets_diff} ({wickets})")
            except ValueError:
                pass
        return changes
    
    image_folder = "<path_to_frames_folder>"
    changed_frames = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            #text = extract_text_from_image(image_path)
            changes = detect_merge(text)
            if changes:
                changed_frames.append(int(filename[:-4]))
            
            clip_folder = "C:\yolov5AMUG"
            clip_length = 420

        for frame_index in changed_frames:
            start_frame = max(1, frame_index - clip_length)
            end_frame = min(frame_index + clip_length, total_frames)

            command = f"ffmpeg -start_number {start_frame} -i {image_folder}/%d.jpg -vframes {2 * clip_length + 1} {clip_folder}/{frame_index}.mp4"
#            subprocess.call(command, shell=True)

    
        

