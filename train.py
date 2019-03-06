"""
DeepLab v3+ with PyTorch
anchor: Lin Zhongya
E-mail:Danial.linzhy@gmail.com

reference ：
pytorch-deeplab-xception https://github.com/jfzhang95/pytorch-deeplab-xception
deeplab https://github.com/tensorflow/models/tree/master/research/deeplab
"""
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.deeplab import DeepLabV3Plus
from dataloader import data_loader


def get_parser():
    parser = argparse.ArgumentParser(description='DeepLab v3+ with PyTorch')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--crop_size', type=int, default=513,
                        help='crop image size')

    # training params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')

    # optimizer params
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--final_lr', default=0.0001, help=
                        'final learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--resume', '-r', action='store_true',
                        default=False, help='resume from checkpoint')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov')

    # cuda
    parser.add_argument('--gpu_ids', type=str, default=[0],
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    return parser


def build_model(args, cuda_device, check=None, num_classes=21):
    print('Building model ...')
    model = DeepLabV3Plus(backbone=args.backbone, sync_bn=False, num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda() if cuda_device else model

    if check:
        model.load_state_dict(check['net'])

    return model


def train(model, train_loader, epoch, device, optimizer, criterion):
    print('training epoch-{}'.format(epoch))
    train_loss = .0
    model.train()
    for i, sample in enumerate(tqdm(train_loader)):
        img, target = sample['image'], sample['label']
        if device:
            img, target = img.cuda(), target.cuda()

        optimizer.zero_grad()
        train_out = model(img)
        loss = criterion(train_out, target.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss


def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = .0
    tbar = tqdm(val_loader, desc='\n')
    for i, sample in enumerate(tqdm(val_loader)):
        img, target = sample['image'], sample['label']
        if device:
            img, target = img.cuda(), target.cuda()

        with torch.no_grad():
            val_out = model(img)

        loss = criterion(val_out, target)
        val_loss += loss.item()

    return val_loss


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = {
            'coco': 30,
            'citvscapes': 300,
            'pascal': 50
        }[args.dataset]

    if args.lr is None:
        args.lr = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }[args.dataset]

    check_name = 'backbone{}-dataset{}-lr{}-momentum{}'.format(
        args.backbone, args.dataset, args.lr, args.momentum)

    if args.resume:
        checkpoint = 'checkpoints/' + check_name
        check = torch.load(checkpoint)
        best_val_loss = check['loss']
        args.start_epoch = check['epoch'] + 1
    else:
        check = None
        best_val_loss = np.inf
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
    cuda_device = torch.cuda.is_available()

    # train and test
    train_loader, val_loader, test_loader, num_classes = data_loader(args)

    # build model
    model = build_model(args, cuda_device, check, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss().cuda() if cuda_device else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8,
                                  patience=5, verbose=True, threshold=0.0001,
                                  threshold_mode='rel', cooldown=3, min_lr=0,
                                  eps=1e-08)

    train_accuracies = []
    val_accuracies = []

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(model, train_loader, epoch, cuda_device, optimizer, criterion)
        val_loss = validate(model, val_loader, cuda_device, criterion)

        scheduler.step(val_loss)
        is_best = val_loss < best_val_loss
        if is_best:
            # Save checkpoint
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model.state_dict(), 'checkpoint/' + check_name + '.pth')
            best_val_loss = val_loss

        # train_accuracies.append(train_loss)
        # val_accuracies.append(val_loss)
        # if not os.path.isdir('curve'):
        #     os.mkdir('curve')
        # torch.save({'train_loss': train_accuracies, 'test_loss': val_accuracies},
        #            os.path.join('curve', check_name))

    # set log, next version should add


if __name__ == '__main__':
    main()

