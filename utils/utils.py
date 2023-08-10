import os
import cv2
import torch
import numpy as np

def fitness_test(preds, targets):
    # IoU, pixel accuracy
    intersection = (preds & targets).float().sum((1, 2))  
    union = (preds | targets).float().sum((1, 2))        
    iou = (intersection + 1e-6) / (union + 1e-6)
    pix_accuracy = (preds == targets).float().sum((1,2))/preds[0].numel()  
    
    return iou, pix_accuracy
     

def plot_image(img, label_img=None, save_file='image.png'):
    # if img is tensor, convert to cv2 image
    if torch.is_tensor(img):
        img = img.mul(255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

    if label_img is not None:
        # if label_img is tensor, convert to cv2 image
        if torch.is_tensor(label_img):
            color_label_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            color_label_img[...,2] = label_img.mul(255.0).cpu().numpy().astype(np.uint8)
            label_img = color_label_img
        # overlay images
        img = cv2.addWeighted(label_img, 0.3, img, 1.0, 0)
    # save image
    cv2.imwrite(save_file, img)

