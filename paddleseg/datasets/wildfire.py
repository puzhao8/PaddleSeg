# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script was adapted from ade.py
import os

import tifffile as tiff

import numpy as np
from PIL import Image

from paddleseg.datasets import Dataset
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F

URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


@manager.DATASETS.add_component
class wildfire(Dataset):
    """
    ADE20K dataset `http://sceneparsing.csail.mit.edu/`.

    Args:
        transforms (list): A list of image transformations.
        dataset_root (str, optional): The ADK20K dataset directory. Default: None.
        mode (str, optional): A subset of the entire dataset. It should be one of ('train', 'val'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 2

    def __init__(self, transforms, dataset_root=None, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge

        if mode not in ['train', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'val') in S1S1 dataset, but got {}."
                .format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        # removed on Jan-05
        # if self.dataset_root is None:
        #     self.dataset_root = download_file_and_uncompress(
        #         url=URL,
        #         savepath=seg_env.DATA_HOME,
        #         extrapath=seg_env.DATA_HOME,
        #         extraname='ADEChallengeData2016')
        # elif not os.path.exists(self.dataset_root):
        #     self.dataset_root = os.path.normpath(self.dataset_root)
        #     savepath, extraname = self.dataset_root.rsplit(
        #         sep=os.path.sep, maxsplit=1)
        #     self.dataset_root = download_file_and_uncompress(
        #         url=URL,
        #         savepath=savepath,
        #         extrapath=savepath,
        #         extraname=extraname)

        if mode == 'train':
            img_dir = os.path.join(self.dataset_root, 'train/S1')
            label_dir = os.path.join(self.dataset_root, 'train/mask/poly')

        elif mode == 'val':
            img_dir = os.path.join(self.dataset_root, 'test/S1')
            label_dir = os.path.join(self.dataset_root, 'test/mask/poly')

        # img_files = os.listdir(img_dir)
        post_dir = os.path.join(img_dir, 'post')
        img_files = os.listdir(post_dir)

        label_files = img_files

        for i in range(len(img_files)):
            # img_path = os.path.join(img_dir, img_files[i])

            pre_path = os.path.join(img_dir, f'pre/{img_files[i]}')
            post_path = os.path.join(img_dir, f'post/{img_files[i]}')

            label_path = os.path.join(label_dir, label_files[i])
            self.file_list.append([(pre_path, post_path), label_path])

    def normalize_sar(self, im):
        return (np.clip(im, -30, 0) + 30) / 30

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        pre_path, post_path = image_path[0], image_path[1]

        if self.mode == 'val':
            # im, _ = self.transforms(im=image_path)
            im1 = tiff.imread(pre_path)
            im2 = tiff.imread(post_path)
            im = np.concatenate((im1, im2), axis=0)
            im = self.normalize_sar(im)

            label = np.asarray(Image.open(label_path))
            # The class 0 is ignored. And it will equal to 255 after
            # subtracted 1, because the dtype of label is uint8.
            # label = label - 1 
            label = label[np.newaxis, :, :]
            return im, label
            
        else:
            # im, label = self.transforms(im=image_path, label=label_path)
            # label = label - 1
            im1 = tiff.imread(pre_path)
            im2 = tiff.imread(post_path)
            im = np.concatenate((im1, im2), axis=0)
            im = self.normalize_sar(im)

            label = np.asarray(Image.open(label_path))

            # Recover the ignore pixels adding by transform
            # label[label == 254] = 255
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im, label, edge_mask
            else:
                return im, label
