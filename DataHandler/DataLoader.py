import os

import cv2
import numpy as np
import torch
from torch import tensor
# from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from DataHandler.DataPreprocessor import DataPreprocessor

class DataLoader:

    def __init__(self,imgfolderpath,labelfolderpath):
        self.datax = self.init_datax(imgfolderpath)
        self.datay = self.init_datay(labelfolderpath)

    def init_datax(self, imgfolderpath):
        listImgFilesPath = os.listdir(imgfolderpath)
        imgs = {}
        for imgFilePath in listImgFilesPath:
            #img = cv2.imread(imgfolderpath + imgFilePath)
            # image_tensor = torch.from_numpy(img).int()
            # img = DataPreprocessor.edge_filtering(img)
            img = DataPreprocessor.read_image(imgfolderpath + imgFilePath)
            # img = DataPreprocessor.apply_ben_preprocessing(img)
            # img = DataPreprocessor.apply_denoising(img)
            # image_tensor = self.format_img(img)
            imgs[imgFilePath[:-4]] = img
        return imgs

    def init_datay(self, labelfolderpath):
        listLabelFile = os.listdir(labelfolderpath)
        datay = {}
        for labelfile in listLabelFile:
            with open(labelfolderpath + labelfile, 'r') as f:
                boxes = []
                for line in f:
                    # label, x_center, y_center, width, height = line.split()
                    stry = line.split()
                    # label, x_center, y_center, width, height = line.split()
                    y_np = np.array(stry[1:], dtype=np.float64)
                    per_line = y_np
                    # per_line = list
                    boxes.append(per_line)
                boxes = np.array(boxes, dtype=np.float64)
                datay[labelfile[:-4]] = (1, torch.from_numpy(boxes))
        return datay

    @staticmethod
    def reformat_img(img):
        """
        :param img: processed img (tensor)
        :return: unprocessed img (numpy)
        """
        image = img * 255
        image = np.array(image.permute(1, 2, 0), dtype='uint8')
        return image

    @staticmethod
    def format_img(img):
        """
        :param img: unprocessed img (numpy )
        :return: processed img (tensor)
        """
        image_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        return image_tensor

    def init_dataset(self):
        dataset = []
        for keyx, valuex in self.datax.items():
            valuey = self.datay[keyx]
            processed_valuey = self.process_datay(valuey, valuex)
            dataset.append((valuex, processed_valuey))
            # reformated_valuex = self.reformat_img(valuex)
            # data = (reformated_valuex, processed_valuey)
            # augmented_dataset = DataPreprocessor.data_augmentation(data)
            # for img, label in augmented_dataset:
            #     img = self.format_img(img)
            #     dataset.append((img, label))
        return dataset

    def process_datay(self, valuey, valuex):
        img_height, img_width, _ = list(valuex.shape)
        boxes = valuey[1]
        x_center = boxes[:,0]
        y_center = boxes[:,1]
        width = boxes[:,2]
        height = boxes[:,3]
        true_width = width * img_width
        true_height = height * img_height
        x_start = x_center*img_width - true_width/2
        y_start = y_center*img_height - true_height/2
        x_end = x_start + true_width
        y_end = y_start + true_height
        new_y = [x_start,y_start,x_end,y_end]
        process_datay = torch.stack(new_y, dim=1)
        return (valuey[0], process_datay)
        # return BoundingBoxes(data=self.datay, format="CXCYWH",canvas_size=canvas_size)

    def init_dataset_permuted(self):
        dataset = []
        for keyx, valuex in self.datax.items():
            # valuex.permute(2,0,1).float()
            valuey = self.datay[keyx]
            dataset.append((valuex, valuey))
        return dataset