import os
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from os2d.modeling.model import build_os2d_from_config
from os2d.config import cfg
import  os2d.utils.visualization as visualizer
from os2d.structures.feature_map import FeatureMapSize
from os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio

logger = setup_logger("OS2D")


# use GPU if have available
cfg.is_cuda = torch.cuda.is_available()

cfg.init.model = "weights/os2d_v2-train.pth"
net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)

input_image0 = read_image("data/demo/input_image.jpg")
input_image = read_image("/media/xin/work1/github_pro/os2d/data/mydata/search/bin/bin5.jpg")
class_images0 = [read_image("data/demo/class_image_0.jpg"),
                read_image("data/demo/class_image_1.jpg")]
class_images = [read_image("/media/xin/work1/github_pro/os2d/data/mydata/template/bin/bin.jpg")]
class_ids = [0]

transform_image = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(img_normalization["mean"], img_normalization["std"])
                      ])


h, w = get_image_size_after_resize_preserving_aspect_ratio(h=input_image.size[1],
                                                               w=input_image.size[0],
                                                               target_size=1500)
input_image = input_image.resize((w, h))

input_image_th = transform_image(input_image)
input_image_th = input_image_th.unsqueeze(0)
if cfg.is_cuda:
    input_image_th = input_image_th.cuda()


class_images_th = []
for class_image in class_images:
    h, w = get_image_size_after_resize_preserving_aspect_ratio(h=class_image.size[1],
                                                               w=class_image.size[0],
                                                               target_size=cfg.model.class_image_size)
    class_image = class_image.resize((w, h))

    class_image_th = transform_image(class_image)
    if cfg.is_cuda:
        class_image_th = class_image_th.cuda()

    class_images_th.append(class_image_th)

with torch.no_grad():
    loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(images=input_image_th,
                                                                                            class_images=class_images_th)

image_loc_scores_pyramid = [loc_prediction_batch[0]]
image_class_scores_pyramid = [class_prediction_batch[0]]
img_size_pyramid = [FeatureMapSize(img=input_image_th)]
transform_corners_pyramid = [transform_corners_batch[0]]

boxes = box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                           img_size_pyramid, class_ids,
                                           nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                           nms_score_threshold=cfg.eval.nms_score_threshold,
                                           # nms_score_threshold=0.45,
                                           transform_corners_pyramid=transform_corners_pyramid)

# remove some fields to lighten visualization
boxes.remove_field("default_boxes")

# Note that the system outputs the correaltions that lie in the [-1, 1] segment as the detection scores (the higher the better the detection).
scores = boxes.get_field("scores")


figsize = (8, 8)
fig=plt.figure(figsize=figsize)
columns = len(class_images)
for i, class_image in enumerate(class_images):
    fig.add_subplot(1, columns, i + 1)
    plt.imshow(class_image)
    plt.axis('off')


plt.rcParams["figure.figsize"] = figsize

cfg.visualization.eval.max_detections = 8
cfg.visualization.eval.score_threshold = float("-inf")
visualizer.show_detections(boxes, input_image,
                           cfg.visualization.eval)









