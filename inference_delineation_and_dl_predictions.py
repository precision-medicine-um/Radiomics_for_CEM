import glob
import os
import argparse
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import keras
import pandas as pd
from keras_maskrcnn import models
from keras_retinanet.utils.image import preprocess_image, resize_image

from keras_maskrcnn.bin.train import create_models
from preprocessing.generate_jpg import *
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

keras.backend.tensorflow_backend._get_available_gpus()
import numpy as np
import ntpath
from tqdm import tqdm

def detect_and_classify_lesions(path_input_recombined, path_input_low_energy, model):
    img_bgr = preprocessing(path_input_low_energy, path_input_recombined)

    mask_prediction = []
    dict_prediction = {}
    labels_to_names = {0: 'benign', 1: 'malignant'}
    image = img_bgr.copy()
    image = preprocess_image(image)
    image, scale = resize_image(image)

    outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes = outputs[-4][0]
    scores = outputs[-3][0]
    labels = outputs[-2][0]
    masks = outputs[-1][0]
    boxes /= scale

    selected_indices = tf.image.non_max_suppression(tf.constant(boxes.reshape(-1, 4)), tf.constant(scores.flatten()),
                                                    max_output_size=tf.constant(5), iou_threshold=0.01)
    selected_boxes = tf.gather(tf.constant(boxes.reshape(-1, 4)), selected_indices)
    session = tf.Session()
    with session.as_default():
        boxes = selected_boxes.eval()

    for box, score, label, mask in zip(boxes, scores, labels, masks):
        if score > 0.1 and box[0] > 0 and box[1] > 0 and box[2] > 0 and box[3] > 0:
            b = box.astype(int)
            dict_prediction[str(b)] = labels_to_names[label]

            temp_mask = mask[:, :,
                        label]  ##warning: the mask return is within the bounding box and has a fix size of 28x28
            mask_prediction.append(temp_mask)

    return np.array(mask_prediction), dict_prediction


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference code for predicting mask and class label with Mask RCNN')
    parser.add_argument('--path_csv', help='Path to the csv containing CEM low energy path and matching CEM '
                                               'recombined path')
    parser.add_argument('--path_dir_to_save', help='Path to the directory to save the results')
    parser.add_argument('--path_dir_weights', help='Path to the weights obtained with train_delineation_model.py')
    args = parser.parse_args()

    ##define paths
    df_paths = pd.read_csv(args.path_csv)
    list_paths_low_energy = list(df_paths["path_low_energy"])
    list_paths_recombined = list(df_paths["path_recombined"])

    ##initialize model
    backbone = models.backbone("resnet101")
    model, _, _ = create_models(backbone_retinanet=backbone.maskrcnn, num_classes=2,
                             weights=args.path_dir_weights, freeze_backbone=False)

    ##predict outcome
    for i, temp_le_path in tqdm(enumerate(list_paths_low_energy)):
        temp_re_path = list_paths_recombined[i]
        mask_prediction, dict_prediction = detect_and_classify_lesions(temp_re_path, temp_le_path,
                                                                       model)
        file = open(args.path_dir_to_save + "/" + ntpath.basename(temp_le_path).split(".")[0]+".pkl", "wb")
        pickle.dump([mask_prediction, dict_prediction], file)
        file.close()

    print("DL predictions generated for " + str(i+1) + " images as pickle")
