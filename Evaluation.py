import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model import Model
from data import getTrainGenerator
from data import getValidationGenerator
from edge_hold_loss import EdgeHoldLoss
import math
import time
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pytorch version of cvpr2019_Pyramid-Feature-Attention-Network-for-Saliency-detection')
    parser.add_argument('--train_file', default='train.txt', help='my train file', type=str)
    parser.add_argument('--test_file', default='test.txt', help='my test file', type=str)
    parser.add_argument('--model_weights', default='model/vgg16_saliencywith1.12saliencypose.pth', help='my model weights', type=str)
    parser.add_argument('--log_interval', default=100, help='step interval between showing logs', type=int)
    parser.add_argument('--save_interval', default=1, help='epoch interval between saving model', type=int)
    parser.add_argument('--pretrained', default=True, help='whether load pretrained weights')

    args = parser.parse_args()
    model_name = args.model_weights
    train_path = args.train_file
    test_path = args.test_file
    print("train_file:", train_path)
    print("test_file:", test_path)
    print("model_weights:", model_name)

    target_size = (256, 256)
    batch_size = 8  # 20
    threshold = 0.5

    f = open(test_path, 'r')
    testlist = f.readlines()
    f.close()

    test_steps = len(testlist) // batch_size
    if len(testlist) % batch_size != 0:
        test_steps += 1

    dropout = True
    with_CA = True
    with_SA = True

    model = Model(dropout=dropout, with_CA=with_CA, with_SA=with_SA)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    if args.pretrained:
        model.load_state_dict(torch.load('model/vgg_pretrained_conv_dict.pth'), strict=False)