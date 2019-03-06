import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
import numpy as np
from tqdm import trange
from pycocotools.coco import COCO
from pycocotools import mask
from PIL import Image, ImageFile

from dataloader import img_transform as tr

ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCOSegmentation(Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self, args, base_dir='datasets/coco',
                 split='train', year='2017'):
        super().__init__()
        self.split = split
        self.args = args
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}'
                                .format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'
                                .format(split, year))
        self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self.preprocess(ids, ids_file)

    def __getitem__(self, item):
        img, target = self.img_gt_point_pair(item)
        sample = {'image': img, 'label': target}
        if self.split == 'train':
            return self.transform_train(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def preprocess(self, ids, ids_file):
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            coco_target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_meta = self.coco.loadImgs(img_id)[0]
            mask = self.get_seg_mask(coco_target, img_meta['height'],
                                     img_meta['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def img_gt_point_pair(self, item):
        coco = self.coco
        img_id = self.ids[item]
        img_meta = coco.loadImgs(img_id)[0]
        path = img_meta['file_name']
        img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        coco_target = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        target = Image.fromarray(self.get_seg_mask(coco_target,
                                 img_meta['height'], img_meta['width']))
        return img, target

    def get_seg_mask(self, target, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], height, width)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomSizedCrop(self.args.crop_size),
            tr.ToTensor(),
            tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.ToTensor(),
            tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)


