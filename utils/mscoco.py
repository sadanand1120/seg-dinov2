# Code from https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/data/dataloader/mscoco.py
"""MSCOCO Semantic Segmentation pretraining for VOC."""
import os
import pickle
import torch
import numpy as np

from tqdm import trange
from PIL import Image, ImageOps
from .segbase import SegmentationDataset


class COCOSegmentation(SegmentationDataset):
    """COCO Semantic Segmentation Dataset for VOC Pre-training.
    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/coco'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = COCOSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    CAT_LIST = [0, 1, 2]
    NUM_CLASS = 3

    def __init__(self, root='/robodata/smodak/corl_rebuttal/dino_traindata/safety', split='train', mode="train", transform=None, **kwargs):
        super(COCOSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # lazy import pycocotools
        from pycocotools.coco import COCO
        from pycocotools import mask
        if split == 'train30':
            print('train30 set')
            ann_file = os.path.join(root, 'train30/mscoco_annotations.json')
            ids_file = os.path.join(root, 'train30/mscoco_ids.mx')
            self.root = os.path.join(root, 'train30/images')
        elif split == 'train':
            print('train set')
            ann_file = os.path.join(root, 'train/mscoco_annotations.json')
            ids_file = os.path.join(root, 'train/mscoco_ids.mx')
            self.root = os.path.join(root, 'train/images')
        elif split == 'val':
            print('val set')
            ann_file = os.path.join(root, 'eval/mscoco_annotations.json')
            ids_file = os.path.join(root, 'eval/mscoco_ids.mx')
            self.root = os.path.join(root, 'eval/images')
        else:
            print('test set')
            ann_file = os.path.join(root, 'test/mscoco_annotations.json')
            ids_file = os.path.join(root, 'test/mscoco_ids.mx')
            self.root = os.path.join(root, 'test/images')
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                self.ids = pickle.load(f)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))
        padding = (0, 210, 0, 210)  # left, top, right, bottom
        img = ImageOps.expand(img, padding, fill=0)
        mask = ImageOps.expand(mask, padding, fill=0)
        img = img.resize((540, 540), Image.BILINEAR)
        mask = mask.resize((540, 540), Image.NEAREST)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(str(self.ids[index]))

    def __len__(self):
        return len(self.ids)

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            # rle = coco_mask.frPyObjects(instance['segmentation'], h, w)  # use this line if segmentation in polygon format
            rle = instance['segmentation']
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

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while." +
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'.
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids

    @property
    def classes(self):
        """Category names."""
        return ('background', 'safe', 'unsafe')
