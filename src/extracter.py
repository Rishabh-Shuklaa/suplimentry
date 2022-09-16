!pip install fid-score
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
import sys
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torchvision
from fid_score import *
fid = FidScore(paths, device, batch_size)
score = fid.calculate_fid_score()
paths = ['/home/user/Desktop/SinGAN-master/now/a', '/home/user/Desktop/SinGAN-master/now/b']
device = torch.device('cuda:0')
batch_size = 1
import cv2
import numpy as np
def extract_frames(path1, path2):
  cap = cv2.VideoCapture(path1)
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
  i = 0
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
      cv2.imwrite(path2 + str(i) + ".png", frame)
      i += 1
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    else: 
      break
  cap.release()  
  cv2.destroyAllWindows()
extract_frames('/home/user/Desktop/phd_work/now/a.mp4', "/home/user/Desktop/phd_work/now/output/")
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 
cfg.MODEL.ROI_HEADS.NMS = 0.8  
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
model = DefaultPredictor(cfg)

from detectron2.utils.visualizer import Visualizer
def plot_bb(path):
  im = cv2.imread(path)
  outputs = model(im)
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2_imshow(v.get_image()[:, :, ::-1])

def crop_class(model, path, clas=0):
  im = cv2.imread(path)
  outputs = model(im)
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2_imshow(v.get_image()[:, :, ::-1])
  with torch.no_grad():
      detections = model(im)["instances"]
  binary_mask = np.zeros((im.shape[0],im.shape[1],im.shape[2]))

  if len(detections.scores)>0:
    index_class = np.where(detections.pred_classes.cpu().numpy()==clas)[0] 
    if len(index_class)>0:
      print(len(index_class), class "detected")
      crop=0
      for i in range(len(index_class)):
        [x1,y1,x2,y2] = detections.pred_boxes[index_class][i].tensor[0].cpu().numpy()
        if y2>150 or (y2<70 and y2>50): 
          crop+=1
          [x1,y1,x2,y2] = [int(x1),int(y1),int(x2)+1,int(y2)+1]
          im[y1:y2, x1:x2, :] = 255
          binary_mask[y1:y2, x1:x2, :] = 1
      print(crop, clas, "cropped")
      
      cv2.imwrite(path.split(".")[0]+"a."+path.split("/")[-1].split(".")[1], im)
      plt.imsave(path[:-(len(path.split("/")[-2])+len(path.split("/")[-1]))-1]+"OUT/"+path.split("/")[-1].split(".")[0]+"mask."+path.split("/")[-1].split(".")[1], binary_mask, dpi=1000)
  return(im, binary_mask, index_class, detections)

plot_bb("/home/user/Desktop/phd_work/now/Input/Images/Bekeleframe0.png" )

plot_bb("/home/user/Desktop/phd_work/now/Fed.png")

plot_bb("/home/user/Desktop/phd_work/now/Boltframe1.png")

plot_bb("/home/user/Desktop/phd_work/now/Nadalframe0.png")

from detectron2.utils.visualizer import Visualizer
def crop_humans(model, path):
  im = cv2.imread(path)
  outputs = model(im)
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2_imshow(v.get_image()[:, :, ::-1])
  with torch.no_grad():
      detections = model(im)["instances"]
  binary_mask = np.zeros((im.shape[0],im.shape[1],im.shape[2]))
  if len(detections.scores)>0:
    index_humans = np.where(detections.pred_classes.cpu().numpy()==0)[0] 
    if len(index_humans)>0:
      print(len(index_humans), "human detected")
      crop=0
      for i in range(len(index_humans)):
        [x1,y1,x2,y2] = detections.pred_boxes[index_humans][i].tensor[0].cpu().numpy()
        if x1>20 and x2<270 and y2>50:
          crop+=1
          [x1,y1,x2,y2] = [int(x1),int(y1),int(x2)+1,int(y2)+1]
          im[y1:y2, x1:x2, :] = 255
          binary_mask[y1:y2, x1:x2, :] = 1
      print(crop, "humans cropped")
      cv2.imwrite(path.split(".")[0]+"_autod."+path.split("/")[-1].split(".")[1], im)
      plt.imsave(path[:-(len(path.split("/")[-2])+len(path.split("/")[-1]))-1]+"Inpainting/"+path.split("/")[-1].split(".")[0]+"_autod_mask."+path.split("/")[-1].split(".")[1], binary_mask, dpi=1000)
  return(im, binary_mask, index_humans, detections)

for i in range(140):
  im, binary_mask, index_humans, detections = crop_humans(model, path = "/home/user/Desktop/phd_work/now/Input/Images/frame"+str(i)+".png")

import cv2
w = 384
h = 288
out = cv2.VideoWriter('/home/user/Desktop/phd_work/now/w.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (w,h))

for i in range(185): 
  im = cv2.imread("/home/user/Desktop/phd_work/now/Boltframe%d.png" %i)
  outputs = model(im)
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  img = v.get_image()[:, :, ::-1]
  cv2.imwrite('/home/user/Desktop/phd_work/now/Boltframe%dbox.png'%i, img)

for i in range(185): 
  img = cv2.imread('/home/user/Desktop/phd_work/now/Boltframe%dbox.png'%i)
  out.write(img)

cv2.destroyAllWindows()
out.release()

from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
import torch
import cv2

if __name__ == '__main__':
    parser = get_arguments("")
    parser.add_argument('--input_dir', help='input image dir', default='/home/user/Desktop/phd_work/SinGAN-master/images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='/home/user/Desktop/phd_work/SinGAN-master/Inpainting')
    parser.add_argument('--inpainting_start_scale', help='inpainting injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='inpainting')
    parser.add_argument('--radius', help='radius harmonization', type=int, default = 10)
    parser.add_argument('--multiple_holes', help = 'set true for multiple holes', action = "store_true")
    parser.add_argument('--ref_name', help='training image name', type = str, default = "")
    """parser.add_argument('--hl', default=0, type=int)
    parser.add_argument('--hh', default=0, type=int)
    parser.add_argument('--wl', default=0, type=int)
    parser.add_argument('--wh', default=0, type=int)"""
    opt = parser.parse_args("")
    opt = functions.post_config(opt)
    if opt.ref_name =="":
        opt.ref_name = opt.input_name
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.inpainting_start_scale < 1) | (opt.inpainting_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            m = cv2.imread('%s/%s_mask%s' % (opt.ref_dir, opt.ref_name[:-4], opt.ref_name[-4:]))
            m = 1 - m / 255
            img = cv2.imread('%s/%s' % (opt.input_dir, opt.ref_name))

            if not opt.multiple_holes:
                positions = np.where(m == 0)
                h_low = positions[0][0]
                h_high = positions[0][-1]
                w_low = positions[1][0]
                w_high = positions[1][-1]

            for j in range(3):
                window = 10
                if not opt.multiple_holes:
                    img[:, :, j][m[:, :, j] == 0] = img[max(h_low - window, 0):min(h_high + window, m.shape[0]),
                                                    max(w_low - window, 0):min(w_high + window, m.shape[1]), j][
                        m[max(h_low - window, 0):min(h_high + window, m.shape[0]),
                        max(w_low - window, 0):min(w_high + window, m.shape[1]), j] == 1].mean()
                else:
                    img[:, :, j][m[:, :, j] == 0] = img[:,:, j][m[:,:, j] == 1].mean()

            cv2.imwrite('%s/%s_averaged%s' % (opt.input_dir, opt.ref_name[:-4], opt.ref_name[-4:]), img)

            ref = functions.read_image_dir('%s/%s_averaged%s' % (opt.input_dir, opt.ref_name[:-4], opt.ref_name[-4:]), opt)
            mask = functions.read_image_dir('%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:]), opt)

            if ref.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]
            mask = functions.dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.inpainting_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            out = (1-mask)*real+mask*out
            plt.imsave('%s/start_scale=%d.png' % (dir2save,opt.inpainting_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)

from skimage import measure
import argparse
import imutils
import cv2
ap = argparse.ArgumentParser("")
ap.add_argument("-f", "--first", required=True, help="Directory of the image that will be compared", default = "/home/user/Desktop/phd_work/now/a.png")
ap.add_argument("-s", "--second", required=True, help="Directory of the image that will be used to compare", default="/home/user/Desktop/phd_work/now/SinGAN1.png")
args = vars(ap.parse_args(""))

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


(score, diff) = measure.compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

PSNR = cv2.PSNR(imageA, imageB)

print("SSIM: {}".format(score), "PSNR: {}".format(PSNR))
