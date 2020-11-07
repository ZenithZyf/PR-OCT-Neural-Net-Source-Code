import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2
import scipy.io as sio

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

# from dataloader import CocoDataset, CSVDataset, collater,\
# Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer

from dataloader import CSVDataset, collater,\
Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

dataset_val = CSVDataset(train_file='val_csv/imgm_Patient3_Normal_test.csv', class_list='octID.csv',\
                         transform=transforms.Compose([Normalizer(), Resizer()]))

sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

retinanet = torch.load('Results_Jan24_2019/model_Jan24.pt')

use_gpu = True

if use_gpu:
    retinanet = retinanet.cuda()

retinanet.eval()

unnormalize = UnNormalizer()

print(len(dataloader_val))


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


for idx, data in enumerate(dataloader_val):

    with torch.no_grad():
        st = time.time()
        scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
        # print('Elapsed time: {}'.format(time.time()-st))
        idxs = np.where(scores > 0.5)
        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

        img[img < 0] = 0
        img[img > 255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        print('Img_ind:',idx)


        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = dataset_val.labels[int(classification[idxs[0][j]])]
            if label_name == 'teeth':
                draw_caption(img, (x1, y1, x2, y2), label_name)

            mycolor = [0, 0, 255]

            #if label_name == 'glare':
            #    mycolor = [255, 0, 0]
            if label_name == 'teeth':
                mycolor = [0, 255, 0]

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(mycolor), thickness=2)
            # print(label_name)

        if idx == 0:
            imgStore = np.expand_dims(img,axis=0)
        else:
            imgStore = np.concatenate((imgStore,np.expand_dims(img,axis=0)),axis=0)

mdict = {'imgStore': imgStore}
sio.savemat('imgStore.mat', mdict)




