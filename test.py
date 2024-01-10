import glob
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
from utils import file


def get_input_image(transform_image,input_image):
    h, w = get_image_size_after_resize_preserving_aspect_ratio(h=input_image.size[1],
                                                               w=input_image.size[0],
                                                               target_size=1500)
    input_image = input_image.resize((w, h))

    input_image_th = transform_image(input_image)
    input_image_th = input_image_th.unsqueeze(0)
    if cfg.is_cuda:
        input_image_th = input_image_th.cuda()
    return input_image_th,input_image

def get_class_images(transform_image,class_images,cfg):
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

    return class_images_th



def get_model_pre(net,input_image_th,class_images_th,box_coder,class_ids,cfg):
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
    return boxes

def show_res(boxes, input_image,class_images,savefig):
    figsize = (8, 8)
    fig = plt.figure(figsize=figsize)
    columns = len(class_images)
    for i, class_image in enumerate(class_images):
        fig.add_subplot(1, columns, i + 1)
        plt.imshow(class_image)
        plt.axis('off')

    plt.rcParams["figure.figsize"] = figsize

    cfg.visualization.eval.max_detections = 8
    cfg.visualization.eval.score_threshold = float("-inf")
    visualizer.show_detections(boxes, input_image,
                               cfg.visualization.eval,savefig=savefig)


def main_test(input_image,class_images,save_fig):
    logger = setup_logger("OS2D")
    cfg.is_cuda = torch.cuda.is_available()
    cfg.init.model = "weights/os2d_v2-train.pth"
    net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(img_normalization["mean"], img_normalization["std"])
    ])
    # input_image0 = read_image("data/demo/input_image.jpg")
    # class_images0 = [read_image("data/demo/class_image_0.jpg"),
    #                  read_image("data/demo/class_image_1.jpg")]
    class_ids = list(range(len(class_images)))
    input_image_th,input_image = get_input_image(transform_image,input_image)
    class_images_th = get_class_images(transform_image,class_images, cfg)
    boxs = get_model_pre(net, input_image_th, class_images_th, box_coder, class_ids, cfg)
    show_res(boxs,input_image,class_images,save_fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='template match')
    parser.add_argument("--image1", type=str, help="seach image path")
    parser.add_argument("--image2", type=str, help="template image path")
    parser.add_argument("--save_path", type=str, default="test_res", help="test image save path")
    args = parser.parse_args()
    search = file.Walk(args.image1, ["png", "jpeg", "jpg"])
    listing = [glob.glob(s)[0] for s in search]
    class_images = [read_image(args.image2)]
    if args.save_path:
        image_name = os.path.join(args.save_path,os.path.basename(args.image1))
        os.makedirs(image_name, exist_ok=True)
    for im in listing:
        input_image = read_image(im)
        if args.save_path:
            save_fig = os.path.join(image_name, os.path.basename(im))
        else:
            save_fig = ""
        main_test(input_image,class_images,save_fig)



