import torch
import numpy as np

from .models import FCN, save_model, load_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, EarlySaveModel
from . import dense_transforms as transforms
import torch.utils.tensorboard as tb
from torchvision import transforms as T

def train(args):
    from os import path

    # hyper parameter
    n_epochs = int(args.epoch)
    batch_size = int(args.batch)
    lr = float(args.learning)

    print( 'epoch: ', n_epochs, 'batch_size: ', batch_size, 'lr: ', lr)

     # device
    if torch.cuda.is_available():
        u_device = 'cuda'
    elif torch.backends.mps.is_available():
        u_device = 'mps'
    else:
        u_device = 'cpu'

    device = torch.device(u_device)
    print('Current device ', device)
    
    # Model
    load_previous_model = bool(args.loadModel)
    if load_previous_model:
        model = load_model('fcn').to(device)
    else:
        model = FCN().to(device)

    model.train()

    # Loss
    weights = torch.tensor(DENSE_CLASS_DISTRIBUTION).to(device=device)
    loss = torch.nn.CrossEntropyLoss()

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr= lr, betas=(0.9, 0.999), weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

     # training scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=20)

    # Early model saver
    acc = float(args.start_accuracy)
    model_saver = EarlySaveModel(acc)

        # data transforms
    data_transforms = {
        'train': transforms.Compose([
            # transforms.ColorJitter(brightness=.1, hue=.2),
            transforms.ColorJitter(brightness=(0.2,1.8),contrast=(0.2, 1.9),saturation=(0.2,2.0),hue=(-0.1,0.1)),
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(128),
            # transforms.RandomResizedCrop(64),
            # transforms.RandomHorizontalFlip(),
            # T.RandomRotation(20),
            T.RandomAutocontrast(),
            T.RandomGrayscale(),
            T.RandomErasing(),
            T.RandomAdjustSharpness(sharpness_factor=0.4),
            T.RandomInvert(),
            # transforms.RandomEqualize(0.5),
            # transforms.RandomSolarize(threshold=100),
            # transforms.RandomPerspective(distortion_scale=0.4, p=1.0),
            # transforms.ColorJitter(brightness=.5, hue=.3),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # # transforms.RandomPosterize(bits=2)
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    # training loop
    global_step = 0
    for epoch in range(n_epochs):
        # training data 
        # num_workers=10
        data_train = load_dense_data('./dense_data/train', num_workers=0, batch_size=batch_size, data_transform=data_transforms['train'] )

        # iterate 
        cm = ConfusionMatrix()
        for batch, labels in data_train:
            batch = batch.to(device)
            labels = labels.long().to(device)

            # Compute loss
            o = model(batch)
            loss_val = loss(o, labels)
            
            # train_logger.add_scalar('loss', loss_val, global_step)
            log(train_logger, batch, labels, logits=o, global_step=global_step)
            cm.add(o.argmax(1), labels)


            # Compute accuracy
            # acc = cm.average_accuracy()
            # train_accuracy.append(accuracy(o, labels).cpu())
            # cm.add(0, labels)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        train_accuracy = cm.average_accuracy.detach().cpu()
        train_iou = cm.iou.detach().cpu()
        train_logger.add_scalar('train/IOU', train_iou, global_step)
        train_logger.add_scalar('train/accuracy', train_accuracy, global_step)

        # Evaluate model
        data_valid = load_dense_data('./dense_data/valid', num_workers=0, batch_size=batch_size, data_transform=data_transforms['val'] )
        
        cm_valid = ConfusionMatrix()
        for v_batch, v_labels in data_valid:
            v_batch = v_batch.to(device)
            v_labels = v_labels.long().to(device)
            valid_pred = model(v_batch)
            
            log(valid_logger, v_batch, v_labels, valid_pred, global_step)
            cm_valid.add(valid_pred.argmax(1), v_labels)

        # mean_valid_accuracy = cm_valid.average_accuracy()
        # valid_logger.add_scalar('valid/accuracy', mean_valid_accuracy, global_step)   
        valid_accuracy = cm_valid.average_accuracy.detach().cpu()
        valid_iou =  cm_valid.iou.detach().cpu()
        valid_logger.add_scalar('valid/iou', valid_iou, global_step)
        valid_logger.add_scalar('valid/accuracy', valid_accuracy, global_step)
        
        # Early save model
        save_state = model_saver.early_save_model(model, valid_iou)
        print('Train iou',train_iou.detach().cpu(), 
              'Train accuracy', train_accuracy.detach().cpu(),
              'Val iou', valid_iou.detach().cpu(), 
              'Val accuracy', valid_accuracy.detach().cpu(),
              'at epoch', epoch, 
              'lr', optimizer.param_groups[0]['lr'], 
              'save', save_state)
        scheduler.step(valid_iou)
        # scheduler.step()

    # save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-ld', '--log_dir', default='./logf')
    parser.add_argument('-e', '--epoch', default=100)
    parser.add_argument('-lr', '--learning', default=0.01)
    parser.add_argument('-lm', '--loadModel', default=0)
    parser.add_argument('-b', '--batch', default=128)
    parser.add_argument('-sa', '--start_accuracy', default=0.0)

    # parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
