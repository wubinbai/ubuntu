### my imports
from matplotlib import pyplot as plt
###
import torch, torchvision
import os

print(torch.__version__, torch.cuda.is_available())
print(os.system('gcc --version'))


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow
from cv2 import imshow
cv2_imshow = imshow
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

jpg_name = '3_families_20140131_140746.jpg'
eg_jpg = '/media/b/TOSHIBA EXT/3_github_data/images/' + jpg_name

im = cv2.imread(eg_jpg)
plt.figure()
#cv2_imshow('three families,', im)
b,g,r = cv2.split(im)
image_rgb = cv2.merge([r,g,b])
plt.ion()
plt.imshow(image_rgb)
plt.show()



######
 
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
im2 = out.get_image()[:, :, ::-1]
b,g,r = cv2.split(im2)
image_rgb2 = cv2.merge([r,g,b])
plt.figure()
plt.imshow(image_rgb2)
plt.show()

