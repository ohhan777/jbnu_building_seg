import os
import cv2
import json
import numpy as np

def plot_image(img_filename, label_filename=None):
    # read image file
    img = cv2.imread(img_filename)
    # read json file
    if label_filename is not None:
        # read json file
        labels = read_json(label_filename)
        # convert json to segmentation mask
        # draw segmentation masks
        h, w, c = img.shape
        label_img = np.zeros((h, w, c), dtype=np.uint8)
        for label in labels:
            pos = np.array(label[0], dtype=np.int32).reshape(-1, 2)
            label_img = cv2.fillPoly(label_img, [pos], (0, 0, 255))

        # overlay images
        img = cv2.addWeighted(label_img, 0.3, img, 1.0, 0)
        
    # save image
    save_filename = os.path.basename(img_filename).replace('.png', '_overlay.png' if label_filename is not None else '.png')
    cv2.imwrite(save_filename, img)


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
    img_filename = './data/kari_building_v1.5/train/images/BLD00001_PS3_K3A_NIA0276.png'
    label_filename = img_filename.replace('images/', 'labels/').replace('.png', '.json')
    plot_image(img_filename, label_filename)
    