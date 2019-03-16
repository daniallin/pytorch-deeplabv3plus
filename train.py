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
from utils.evaluate import Evaluator
from utils.keeper import Keeper


def get_parser():
    parser = argparse.ArgumentParser(description='DeepLab v3+ with PyTorch')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name')
    parser.add_argument('--workers', type=int, default=4,
                        help='dataloader threads')
    parser.add_argument('--crop_size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters')
    parser.add_argument('--output-scale', type=int, default=16,
                        help='network output stride')

    # training params
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # optimizer params
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--final_lr', default=0.0001, help=
                        'final learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--resume', '-r', default=None, type=str,
                        help='resume from checkpoint')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov')

    # cuda
    parser.add_argument('--gpu_ids', type=str, default=[0],
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    return parser


def train(model, train_loader, epoch, cuda_device, optimizer, criterion):
    print('training epoch-{}'.format(epoch))
    train_loss = .0
    num_train_img = len(train_loader)
    model.train()
    prog_bar = tqdm(train_loader, miniters=5, desc='\n')
    for i, sample in enumerate(prog_bar):
        img, target = sample['image'], sample['label']
        if cuda_device:
            img, target = img.cuda(), target.cuda()

        optimizer.zero_grad()
        train_out = model(img)
        loss = criterion(train_out, target.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        prog_bar.set_description('train loss: {}'.format(train_loss/(i+1)))
        # just for testing
        pred = train_out.data.cpu().numpy()
        print(pred.shape)

    return train_loss


def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = .0
    prog_bar = tqdm(val_loader, miniters=5, desc='\n')
    for i, sample in enumerate(prog_bar):
        img, target = sample['image'], sample['label']
        if device:
            img, target = img.cuda(), target.cuda()

        with torch.no_grad():
            val_out = model(img)
        loss = criterion(val_out, target.long())
        val_loss += loss.item()
        prog_bar.set_description('Test loss: {}'.format(val_loss / (i + 1)))
        pred = val_out.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        target = target.cpu().numpy()
        img = img.cpu().numpy()

    return val_loss, img, pred, target


def main(args, cuda_device):
    # train and validation dataloader
    train_loader, val_loader, test_loader, num_classes = data_loader(args)

    # build model
    print('Building model ...')
    model = DeepLabV3Plus(backbone=args.backbone, sync_bn=args.sync_bn,
                          num_classes=num_classes, freeze_bn=args.freeze_bn,
                          output_scale=args.output_scale, pretrained=True)
    model = torch.nn.DataParallel(model).cuda() if cuda_device else model
    criterion = nn.CrossEntropyLoss().cuda() if cuda_device else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8,
                                  patience=5, verbose=True, threshold=0.0001,
                                  threshold_mode='rel', cooldown=3, min_lr=0,
                                  eps=1e-08)

    # Whether using checkpoint
    if args.resume:
        check_path = os.path.join('checkpoints', args.resume)
        if not os.path.exists(check_path):
            raise RuntimeError("=> no checkpoint found")
        checkpoint = torch.load(check_path)
        if cuda_device:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_pred = checkpoint['best_pred']
        args.start_epoch = checkpoint['epoch'] + 1
    else:
        best_pred = .0

    print(args)

    evaluator = Evaluator(num_classes)
    keeper = Keeper(args)
    keeper.save_experiment_config()

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(model, train_loader, epoch, cuda_device, optimizer, criterion)
        val_loss, img, prediction, target = validate(model, val_loader, cuda_device, criterion)

        evaluator.reset()
        evaluator.add_batch(target, prediction)
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        mIoU = evaluator.Mean_Intersection_over_Union()

        scheduler.step(val_loss)
        is_best = mIoU > best_pred
        if is_best:
            best_pred = mIoU
            # Save checkpoint
            print('Saving..')
            keeper.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
            })

        keeper.save_train_log({
            'current_epoch': epoch,
            'train_loss': train_loss,
            'validate_loss': val_loss,
            'mIoU': mIoU
            })

        if epoch % 10 == 0:
            keeper.save_val_img(img, prediction, target, epoch)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
    cuda_device = torch.cuda.is_available()

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
    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)

    main(args, cuda_device)

