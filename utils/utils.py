
def fitness_test(preds, targets):
    # IoU, pixel accuracy
    intersection = (preds & targets).float().sum((1, 2))  
    union = (preds | targets).float().sum((1, 2))        
    iou = (intersection + 1e-6) / (union + 1e-6)
    pix_accuracy = (preds == targets).float().sum((1,2))/preds[0].numel()  
    
    return iou, pix_accuracy
     