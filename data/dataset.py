from os import listdir
from os.path import join
import random

from PIL import Image, ImageFilter
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np

from utils.image_processing import read_split_image
from utils.bytesIO import PickledImageProvider, bytes_to_file


class DatasetFromObj(data.Dataset):
    def __init__(self, obj_path, input_nc=3, augment=False, bold=False, rotate=False, blur=False, start_from=0):
        super(DatasetFromObj, self).__init__()
        self.image_provider = PickledImageProvider(obj_path)
        self.input_nc = input_nc
        if self.input_nc == 1:
            self.transform = transforms.Normalize(0.5, 0.5)
        elif self.input_nc == 3:
            self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            raise ValueError('input_nc should be 1 or 3')
        # sister set -> augment + bold ;
        self.augment = augment
        self.bold = bold
        self.rotate = rotate
        self.blur = blur
        # ? don't know how does this start_from work
        self.start_from = start_from

    def __getitem__(self, index):
        item = self.image_provider.examples[index]
        label = item[0].split('_')[0].split(',')
        style = int(item[0].split('_')[1])
        label = [int(each) for each in label]
        label = torch.tensor(label)
        img_tra_ch, img_sou_seal, img_dst_seal = self.process(item[1])
        return label, img_tra_ch, img_sou_seal, img_dst_seal, style

    def __len__(self):
        return len(self.image_provider.examples)

    def process(self, img_bytes):
        """
            process byte stream to training data entry
        """
        # sister set -> augment + bold ;
        image_file = bytes_to_file(img_bytes)
        img = Image.open(image_file)
        try:
            # tra_ch, sou_seal, dst_seal
            img_tra_ch, img_sou_seal, img_dst_seal = read_split_image(img)
            if self.augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h = img_tra_ch.size
                if self.bold:
                    multiplier = random.uniform(1.0, 1.2)
                else:
                    multiplier = random.uniform(1.0, 1.05)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1

                # Used to use Image.BICUBIC, change to ANTIALIAS, get better image.
                img_tra_ch = img_tra_ch.resize((nw, nh), Image.ANTIALIAS)
                img_sou_seal = img_sou_seal.resize((nw, nh), Image.ANTIALIAS)
                img_dst_seal = img_dst_seal.resize((nw, nh), Image.ANTIALIAS)

                shift_x = random.randint(0, max(nw - w - 1, 0))
                shift_y = random.randint(0, max(nh - h - 1, 0))

                img_tra_ch = img_tra_ch.crop((shift_x, shift_y, shift_x + w, shift_y + h))
                img_sou_seal = img_sou_seal.crop((shift_x, shift_y, shift_x + w, shift_y + h))
                img_dst_seal = img_dst_seal.crop((shift_x, shift_y, shift_x + w, shift_y + h))

                if self.rotate and random.random() > 0.9:
                    angle_list = [0, 180]
                    random_angle = random.choice(angle_list)
                    if self.input_nc == 3:
                        fill_color = (255, 255, 255)
                    else:
                        fill_color = 255
                    img_tra_ch = img_tra_ch.rotate(random_angle, resample=Image.BILINEAR, fillcolor=fill_color)
                    img_sou_seal = img_sou_seal.rotate(random_angle, resample=Image.BILINEAR, fillcolor=fill_color)
                    img_dst_seal = img_dst_seal.rotate(random_angle, resample=Image.BILINEAR, fillcolor=fill_color)

                if self.blur and random.random() > 0.8:
                    sigma_list = [1, 1.5, 2]
                    sigma = random.choice(sigma_list)
                    img_tra_ch = img_tra_ch.filter(ImageFilter.GaussianBlur(radius=sigma))
                    img_sou_seal = img_sou_seal.filter(ImageFilter.GaussianBlur(radius=sigma))
                    img_dst_seal = img_dst_seal.filter(ImageFilter.GaussianBlur(radius=sigma))

                img_tra_ch = transforms.ToTensor()(img_tra_ch)
                img_sou_seal = transforms.ToTensor()(img_sou_seal)
                img_dst_seal = transforms.ToTensor()(img_dst_seal)

                '''
                Used to resize here. Change it before rotate and blur.
                w_offset = random.randint(0, max(0, nh - h - 1))
                h_offset = random.randint(0, max(0, nh - h - 1))

                img_A = img_A[:, h_offset: h_offset + h, w_offset: w_offset + h]
                img_B = img_B[:, h_offset: h_offset + h, w_offset: w_offset + h]
                '''

                img_tra_ch = self.transform(img_tra_ch)
                img_sou_seal = self.transform(img_sou_seal)
                img_dst_seal = self.transform(img_dst_seal)
            else:
                img_tra_ch = transforms.ToTensor()(img_tra_ch)
                img_sou_seal = transforms.ToTensor()(img_sou_seal)
                img_dst_seal = transforms.ToTensor()(img_dst_seal)

                img_tra_ch = self.transform(img_tra_ch)
                img_sou_seal = self.transform(img_sou_seal)
                img_dst_seal = self.transform(img_dst_seal)

            # sister concat the pic
            # B, A, stan
            return img_tra_ch, img_sou_seal, img_dst_seal

        finally:
            image_file.close()
