import os
import shutil
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt

from dataloader.utils import decode_segmap


class Keeper(object):

    def __init__(self, args):
        self.args = args
        self.directory = 'checkpoints'
        self.exps = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        exp_id = int(self.exps[-1].split('_')[-1]) + 1 if self.exps else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(exp_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        best_pred = state['best_pred']
        pre_miou = [0.0]
        with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'a+') as f:
            f.write(str(best_pred) + '\n')

        best_pred_path = os.path.join(filename, 'best_pred.txt')
        if os.path.exists(best_pred_path):
            with open(best_pred_path, 'r') as f:
                pre_miou.append(float(f.readlines()[-1]))

        max_miou = max(pre_miou)
        if best_pred > max_miou:
            shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))

    def save_experiment_config(self):
        config_file = os.path.join(self.experiment_dir, 'parameters.txt')
        p = dict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['lr'] = self.args.lr
        p['epoch'] = self.args.epochs
        p['batch_size'] = self.args.batch_size
        p['crop_size'] = self.args.crop_size
        p['weight_decay'] = self.args.weight_decay
        p['nesterov'] = self.args.nesterov

        with open(config_file, 'w') as f:
            for key, val in p.items():
                f.write(key + ':' + str(val) + '\n')

    def save_train_log(self, logging):
        log_file = os.path.join(self.experiment_dir, 'train_log.txt')
        with open(log_file, 'a+') as f:
            if isinstance(logging, dict):
                for key, val in logging.items():
                    f.write(key + ':' + str(val) + '\n')
            else:
                f.write(str(logging) + '\n')

    def save_val_img(self, img, pred, target, serial):
        img_path = os.path.join(self.experiment_dir, 'images')
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        print('img_shape = {}, pred_shape= {}, target_shape = {}'
              .format(img.shape, pred.shape, target.shape))
        for i in range(img.shape[0]):
            tmp_target = np.array(target[i]).astype(np.uint8)
            tmp_pred = np.array(pred[i]).astype(np.uint8)
            target_segmap = decode_segmap(tmp_target, dataset=self.args.dataset)
            pred_segmap = decode_segmap(tmp_pred, dataset=self.args.dataset)
            img_tmp = np.transpose(img[i], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(221)
            plt.imshow(img_tmp)
            plt.subplot(222)
            plt.imshow(target_segmap)
            plt.subplot(223)
            plt.imshow(pred_segmap)
            plt.savefig(os.path.join(img_path, '{}_val_img.png'.format(serial)))
