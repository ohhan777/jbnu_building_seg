import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from utils.utils import plot_image
import argparse
from torchvision import models


def predict(opt):   
    # model  
    model = models.segmentation.deeplabv3_resnet101(num_classes=2)

    # GPU-support
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # load weights
    assert os.path.exists(opt.weight), "no found the model weight"
    checkpoint = torch.load(opt.weight)
    model.load_state_dict(checkpoint['model'])

    # input
    img = cv2.imread(opt.input)
        
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()/255.0 # tensor in [0,1]
    imgs = img.unsqueeze(0) # (1, 3, H, W)
    
    print('predicting...')
    model.eval()
    imgs.to(device)
    with torch.no_grad():
        preds = model(imgs)['out']  # (1, C, H, W)
        preds = torch.argmax(preds, axis=1) # (1, H, W)
        save_file = os.path.join('outputs', os.path.basename(opt.input).replace('.png', '_pred.png'))
        plot_image(imgs[0], preds[0], save_file)     
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='input image')
    parser.add_argument('--weight', '-w', default='weights/ohhan_best.pth',
                        help='weight file path')
    opt = parser.parse_args()

    predict(opt)
