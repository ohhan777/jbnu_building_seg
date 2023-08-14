import os
import cv2
import json
import numpy as np
from utils.utils import plot_image

def plot_image_file(img_file, label_file=None):
    # read image file
    img = cv2.imread(img_file)
    # read json file
    if label_file is not None:
        # read json file
        labels = read_json(label_file)
        # convert json to segmentation mask
        # draw segmentation masks
        h, w, c = img.shape
        label_img = np.zeros((h, w, c), dtype=np.uint8)
        for label in labels:
            pos = np.array(label[0], dtype=np.int32).reshape(-1, 2)
            label_img = cv2.fillPoly(label_img, [pos], (0, 0, 255))
    else:
        label_img = None
        
    # save image
    save_file = os.path.basename(img_file).replace('.png', '_overlay.png' if label_file is not None else '.png')
    plot_image(img, label_img, save_file)


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
    img_file = './data/kari-building/train/images/BLD00001_PS3_K3A_NIA0276.png'
    label_file = img_file.replace('images/', 'labels/').replace('.png', '.json')
    #plot_image_file(img_file)
    plot_image_file(img_file, label_file)
    
    