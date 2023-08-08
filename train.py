import os
import argparse
import torch
from data.kari_building_dataset import KariBuildingDataset
from utils.utils import fitness_test
from loss import ce_loss
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import torch.optim as optim
import time
from pathlib import Path
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


def train(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    name = opt.name
    # tensorboard settings
    log_dir = Path('logs')/name
    tb_writer = SummaryWriter(log_dir=log_dir)
 
    # Train dataset
    train_dataset = KariBuildingDataset('./data', is_train=True)
    # Train dataloader
    num_workers = min([os.cpu_count(), batch_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    # for i, (imgs, targets) in enumerate(train_dataloader):
    #     cv2_imshow(imgs[0], targets[0])
    #     pass # debug checkpoint

    # Validation dataset
    val_dataset = KariBuildingDataset('./data', is_train=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    # Network model
    model = models.segmentation.deeplabv3_resnet101(progress=True, num_classes=2)
    #model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT, progress=True)
    #num_classes = 2 # background, building
    #model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    #model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        

    # GPU-support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:   # multi-GPU
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=3e-4)
      
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=1)

    # loading a weight file (if exists)
    weight_file = Path('weights')/(name + '.pth')
    best_accuracy = 0.0
    start_epoch, end_epoch = (0, epochs)
    if os.path.exists(weight_file):
        checkpoint = torch.load(weight_file)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print('resumed from epoch %d' % start_epoch)

    # training/validation
    for epoch in range(start_epoch, end_epoch):
        print('epoch: %d/%d' % (epoch, end_epoch-1))
        t0 = time.time()
        epoch_loss = train_one_epoch(train_dataloader, model, optimizer, device)
        t1 = time.time()
        print('loss=%.4f (took %.2f sec)' % (epoch_loss, t1-t0))
        lr_scheduler.step(epoch_loss)
        # validation
        val_epoch_loss, val_epoch_iou, val_epoch_pix_accuracy = val_one_epoch(val_dataloader, model, device)
        print('[validation] loss=%.4f, iou=%.4f, pixel accuracy=%.4f' % (val_epoch_loss, val_epoch_iou, val_epoch_pix_accuracy))
        # saving the best status into a weight file
        if val_epoch_pix_accuracy > best_accuracy:
             best_weight_file = Path('weights')/(name + '_best.pth')
             best_accuracy = val_epoch_pix_accuracy
             state = {'model': model.state_dict(), 'epoch': epoch, 'best_accuracy': best_accuracy}
             torch.save(state, best_weight_file)
             print('best accuracy=>saved\n')
        # saving the current status into a weight file
        state = {'model': model.state_dict(), 'epoch': epoch, 'best_accuracy': best_accuracy}
        torch.save(state, weight_file)
        # tensorboard logging
        tb_writer.add_scalar('train_epoch_loss', epoch_loss, epoch)
        tb_writer.add_scalar('val_epoch_loss', val_epoch_loss, epoch)
        tb_writer.add_scalar('val_accuracy', val_epoch_pix_accuracy, epoch)

def train_one_epoch(train_dataloader, model, optimizer, device):
    model.train()
    losses = [] 
    for i, (imgs, targets) in enumerate(train_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)['out']     # forward 
        loss = ce_loss(preds, targets) # calculates the iteration loss  
        optimizer.zero_grad()   # zeros the parameter gradients
        loss.backward()         # backward
        optimizer.step()        # update weights
        print('\t iteration: %d/%d, loss=%.4f' % (i, len(train_dataloader)-1, loss))    
        losses.append(loss.item())
    return torch.tensor(losses).mean().item()


def val_one_epoch(val_dataloader, model, device):
    model.eval()
    losses = []
    iou_sum = 0
    pix_accuracy_sum = 0
    total = 0
    for i, (imgs, targets) in enumerate(val_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            preds = model(imgs)['out']
            loss = ce_loss(preds, targets)
            preds = torch.argmax(preds, axis=1)
            iou, pix_accuracy = fitness_test(preds, targets.long())   
            losses.append(loss.item())
            iou_sum += iou.sum().item()
            pix_accuracy_sum += pix_accuracy.sum().item()
            total += preds.size(0)
            
    avg_loss = torch.tensor(losses).mean().item()
    avg_iou = iou_sum/total
    avg_pix_accuracy = pix_accuracy_sum/total
    return avg_loss, avg_iou, avg_pix_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--name', default='test1', help='name for the run')

    opt = parser.parse_args()

    train(opt)
