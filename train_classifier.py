import torch
import torchvision
from cnn_finetune import make_model
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
import os
import argparse
import time
from data_handler import get_datasets
from training_tools import *


def main(args, best_prec1, device, train_loader, test_loader, model):

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if args.half:
        print('half persicion is used.')
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)


    if args.evaluate:
        print('evalution mode')
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model.th')))
        best_prec1 = validate(test_loader, model, criterion)
        return best_prec1

    if args.pretrained:
        print('evalution of pretrained model')
        args.save_dir='pretrained_models'
        pretrained_model= args.arch +'.th'
        model.load_state_dict(torch.load(os.path.join(args.save_dir, pretrained_model)))
        best_prec1 = validate(test_loader, model, criterion)
        return best_prec1

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('Training {} model'.format(args.arch))
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint(model.state_dict(), filename=os.path.join(args.save_dir, 'checkpoint.th'))
        if is_best:
            save_checkpoint(model.state_dict(), filename=os.path.join(args.save_dir, 'model_resnet18_cifar100.th'))

    return best_prec1

if __name__ == '__main__':
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATASET', type=str, help='Specify dataset')
    commandLineParser.add_argument('MODEL', type=str, help='Specify architecture to train')
    commandLineParser.add_argument('CLASSES', type=int, help='Specify number of classes')
    commandLineParser.add_argument('SIZE', type=int, help='Specify image dimension')
    commandLineParser.add_argument('--loc', type=str, default=None, help='Can give data location')

    inp_args = commandLineParser.parse_args()
    dataset = inp_args.DATASET
    model_name = inp_args.MODEL
    num_classes = inp_args.CLASSES
    inp_size = inp_args.SIZE
    loc = inp_args.loc

    # Load the data
    train_ds, test_ds = get_datasets(dataset, loc)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Get device
    device = get_default_device()

    # Get training arguments
    args = MyArgs(model_name)

    # Load pretrained_model
    model = make_model(model_name, num_classes=num_classes, pretrained=True, input_size=(inp_size,inp_size))
    model.to(device)

    # Train
    best_prec1 = 0
    best_prec1 = main(args, best_prec1, device, train_loader, test_loader)
    print('The lowest error from {} model after {} epochs is {error:.3f}'.format(args.arch,args.epochs,error=100-best_prec1))
