import os
import cv2
import json
import numpy as np
import glob
from utils.utils import plot_image

def conv_json2png(img_dir, label_dir, png_label_dir):
    # make directory
    if not os.path.exists(png_label_dir):
        os.makedirs(png_label_dir)
    # get all image files
    img_files = glob.glob(os.path.join(img_dir, '*.png'))
    for img_file in img_files:
        print(img_file)
        # read image file
        img = cv2.imread(img_file)
        # read label file
        label_filename = img_file.replace('images/', 'labels/').replace('.png', '.json')
         # read json file
        labels = read_json(label_filename)
        # convert json to segmentation mask
        # draw segmentation masks
        h, w, c = img.shape
        label_img = np.zeros((h, w), dtype=np.uint8)
        for label in labels:
            pos = np.array(label[0], dtype=np.int32).reshape(-1, 2)
            label_img = cv2.fillPoly(label_img, [pos], 255)
        # save label image
        save_file = os.path.join(png_label_dir, os.path.basename(img_file).replace('.png', '_label.png'))
        plot_image(label_img, None, save_file)
        


def read_json(label_file):
    labels = []
    with open(label_file, 'r') as f:
        data_dict = json.load(f)
    for x in data_dict['features']:
        obj = x['properties']
        label = []
        l = obj['building_imcoords'].split(',')
        if l[-1] == '':
            l.pop() 
        l = [round(float(i)) for i in l]
        if len(l) != 0 :
            label.append(l)
            label.append(int(obj['type_id']))
            labels.append(label)
    return labels
    

if __name__ == '__main__':
    img_dir = './data/kari_building_v1.5/train/images/'
    label_dir = img_dir.replace('images/', 'labels/')
    png_label_dir = img_dir.replace('images/', 'png_labels/')
    conv_json2png(img_dir, label_dir, png_label_dir)
    