import torch, torchvision
import glob 
import os
import numpy as np
import cv2
import random
import itertools
import pandas as pd
from tqdm import tqdm
import urllib
import json
import PIL.Image as Image

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import ntpath
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt

import argparse

def return_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.EVAL_PERIOD = 500

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    predictor = DefaultPredictor(cfg)

    MetadataCatalog.get("faces_train").set(thing_classes=['maskon', 'maskoff'])
    statement_metadata = MetadataCatalog.get("faces_train")
    return predictor, statement_metadata


def detect_mask(file_path, save_to):
    predictor, statement_metadata = return_cfg()
    im = cv2.imread(file_path)
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    instances.remove('pred_masks')
    v = Visualizer(
        im[:, :, ::-1],
        metadata=statement_metadata, 
        scale=1., 
        instance_mode=ColorMode.IMAGE
    )
    instances = outputs["instances"].to("cpu")
    instances.remove('pred_masks')
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    file_name = ntpath.basename(file_path)
    cv2.imwrite(f'{save_to}/result_{file_name}', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #plt.imshow(result)
    #plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect mask_on and mask_off on people's faces")
    parser.add_argument('file_path',help="path of the images")
    parser.add_argument('save_path',help="path for saving the results")
    args = parser.parse_args()
    detect_mask(args.file_path, args.save_path)