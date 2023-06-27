from .models import CNNClassifier, save_model, load_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy, EarlySaveModel
import torch
import torchvision
from torchvision import transforms
import torch.utils.tensorboard as tb

def train(args):
    from os import path
    # hyper parameter
    n_epochs = int(args.epoch)
    batch_size = int(args.batch)
    lr = float(args.learning)
    
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
        model = load_model('cnn').to(device)
    else:
        # # We are following ResNet architecture, which is to create 3, 3, 6 & 3 layers
        model = CNNClassifier(hidden_layers_count=[3, 1, 1],
                              hidden_layer_channel_size=[64, 128, 256, 512],
                              hidden_layer_strides=[1, 2, 2, 2]).to(device)
        

        # model = CNNClassifier(hidden_layers_count=[3, 3, 6, 3],
        #                       hidden_layer_channel_size=[64, 128, 256, 512],
        #                       hidden_layer_strides=[1, 2, 2, 2])

    # Loss
    loss = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # Logger
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    # training scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=20)

    # Early model saver
    acc = float(args.start_accuracy)
    model_saver = EarlySaveModel(acc)

    # data transforms
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(128),
            # transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAutocontrast(),
            transforms.RandomGrayscale(),
            # transforms.RandomErasing(),
            transforms.RandomAdjustSharpness(sharpness_factor=0.7),
            transforms.RandomInvert(),
            transforms.RandomEqualize(0.5),
            transforms.RandomSolarize(threshold=100),
            # transforms.RandomPerspective(distortion_scale=0.4, p=1.0),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # transforms.RandomPosterize(bits=2)
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # training loop
    global_step = 0
    for epoch in range(n_epochs):
        # training data 
        # num_workers=10
        data_train = load_data('./data/train', batch_size=batch_size, data_transform=data_transforms['train'])

        # iterate 
        train_accuracy = []
        for batch, labels in data_train:
            batch = batch.to(device)
            labels = labels.to(device)

            # Compute loss
            o = model(batch)
            loss_val = loss(o, labels)
            
            train_logger.add_scalar('loss', loss_val, global_step)

            # Compute accuracy
            train_accuracy.append(accuracy(o, labels).cpu())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        mean_train_accuracy = sum(train_accuracy) / len(train_accuracy)
        valid_logger.add_scalar('train/accuracy', mean_train_accuracy, global_step)

        # Evaluate model
        data_valid = load_data('./data/valid', batch_size=batch_size, data_transform=data_transforms['val'])
        raw_accuracy = []
        for v_batch, v_labels in data_valid:
            v_batch = v_batch.to(device)
            v_labels = v_labels.to(device)
            valid_pred = model(v_batch)
            raw_accuracy.append(accuracy(valid_pred, v_labels).cpu())

        
        mean_valid_accuracy = torch.FloatTensor(sum(raw_accuracy) / len(raw_accuracy))
        valid_logger.add_scalar('valid/accuracy', mean_valid_accuracy, global_step)    
        # Early save model
        save_state = model_saver.early_save_model(model, mean_valid_accuracy)
        print('Training Accuracy',mean_train_accuracy, 'Validation Accuracy ', mean_valid_accuracy, 'at epoch', epoch, 'lr', optimizer.param_groups[0]['lr'], 'model save reponse ', save_state)
        scheduler.step(mean_train_accuracy)

    # save model
    # save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-ld', '--log_dir')
    parser.add_argument('-e', '--epoch', default=100)
    parser.add_argument('-lr', '--learning', default=0.01)
    parser.add_argument('-lm', '--loadModel', default=0)
    parser.add_argument('-b', '--batch', default=128)
    parser.add_argument('-sa', '--start_accuracy', default=0.0)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

