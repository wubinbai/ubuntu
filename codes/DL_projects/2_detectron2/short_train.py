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


 
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


#v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#im2 = out.get_image()[:, :, ::-1]
#b,g,r = cv2.split(im2)
#image_rgb2 = cv2.merge([r,g,b])
#plt.figure()
#plt.imshow(image_rgb2)
#plt.show()


############## above is part a, without e.g. jpg

import os

# download, decompress the data
#os.system('wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip')
#os.system('unzip balloon_dataset.zip > /dev/null')

### register the dataset
# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")
### end of register

### original visualize data
#dataset_dicts = get_balloon_dicts("balloon/train")
#for d in random.sample(dataset_dicts, 3):
#    img = cv2.imread(d["file_name"])
#    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
#    out = visualizer.draw_dataset_dict(d)
#    cv2_imshow(out.get_image()[:, :, ::-1])
### end visualize

### my local visualize data
dataset_dicts = get_balloon_dicts("balloon/train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    b,g,r=cv2.split((out.get_image()[:, :, ::-1]))
    image_rgb = cv2.merge([r,g,b])
    plt.figure()
    plt.ion()
    plt.imshow(image_rgb)
    plt.show()
### end visualize

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#cfg.SOLVER.STEPS= (15,280)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

###
#%load_ext tensorboard
#%tensorboard --logdir output
###


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

####

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_balloon_dicts("balloon/val")
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure()

    #cv2_imshow(out.get_image()[:, :, ::-1])
    im_i = out.get_image()[:, :, ::-1]
    b,g,r = cv2.split(im_i)
    image_rgb_i = cv2.merge([r,g,b])
    plt.imshow(image_rgb_i)
    plt.show()


########### evaluate 
########### evaluate 
########### evaluate 
########### evaluate 
########### evaluate 
########### evaluate 
# We can also evaluate its performance using AP metric implemented in COCO API.
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
#evaluator = COCOEvaluator("my_dataset_val", ("bbox", "segm"), False, output_dir="./output/")
evaluator = COCOEvaluator("balloon_val", ("bbox", "segm"), False, output_dir="./output/")
#val_loader = build_detection_test_loader(cfg, "my_dataset_val")
val_loader = build_detection_test_loader(cfg, "balloon_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
########### end od eval
###########
# Inference with a keypoint detection model
#cfg = get_cfg()   # get a fresh new config
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
#predictor = DefaultPredictor(cfg)
#outputs = predictor(im)
#v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#im3 = out.get_image()[:, :, ::-1]
#b,g,r = cv2.split(im3)
#image_rgb3 = cv2.merge([r,g,b])
#plt.figure()
#plt.imshow(image_rgb3)
#plt.show()


####
# Inference with a panoptic segmentation model
#cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
#predictor = DefaultPredictor(cfg)
#panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
#v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

