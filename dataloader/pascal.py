from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader import img_transform as tr


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self, args, base_dir='datasets/VOCdevkit/VOC2012/', split='train'):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super(VOCSegmentation, self).__init__()
        self.base_dir = base_dir
        self.image_dir = os.path.join(self.base_dir, 'JPEGImages')
        self.cat_dir = os.path.join(self.base_dir, 'SegmentationClass')

        self.args = args

        splits_dir = os.path.join(self.base_dir, 'ImageSets', 'Segmentation')
        self.split = [split]
        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                image = os.path.join(self.image_dir, line + ".jpg")
                cat = os.path.join(self.cat_dir, line + ".png")
                assert os.path.isfile(image)
                assert os.path.isfile(cat)
                self.im_ids.append(line)
                self.images.append(image)
                self.categories.append(cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = Image.open(self.categories[index])
        sample = {'image': image, 'label': label}

        for split in self.split:
            if split == "train":
                return self.transform_train(sample)
            elif split == 'val':
                return self.transform_val(sample)

    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomSizedCrop(self.args.crop_size),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        sample = composed_transforms(sample)
        return sample

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.CenterCrop(self.args.crop_size),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloader.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.crop_size = 513

    # voc_train = VOCSegmentation(args, split='train')
    voc_val = VOCSegmentation(args, split='val')

    dataloader = DataLoader(voc_val, batch_size=1, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)
            plt.show()

        if ii == 1:
            break

    plt.show(block=True)


