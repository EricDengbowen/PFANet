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
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lr_scheduler(epoch, base_lr):
     drop = 0.5
     epoch_drop = epochs / 10.
     lr = base_lr * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
     print('lr: %f' % lr)
     return lr




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pytorch version of cvpr2019_Pyramid-Feature-Attention-Network-for-Saliency-detection')
    parser.add_argument('--train_file', default='train.txt', help='my train file', type=str)
    parser.add_argument('--test_file', default='test.txt', help='my test file', type=str)
    parser.add_argument('--model_weights', default='model/vgg16_saliencywith1.12saliencyposeLr1e-3totalalpha0.8.pth', help='my model weights', type=str)
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

    # Model Config
    target_size = (256, 256)
    batch_size = 8    #20
    base_lr = 1e-2
    epochs = 300
    threshold = 0.5

    f = open(train_path, 'r')
    trainlist = f.readlines()
    f.close()

    steps_per_epoch = len(trainlist) // batch_size
    if len(trainlist) % batch_size != 0:
        steps_per_epoch += 1

    f = open(test_path, 'r')
    testlist = f.readlines()
    f.close()

    test_steps = len(testlist) // batch_size
    if len(testlist) % batch_size != 0:
        test_steps += 1

    dropout = True
    with_CA = True
    with_SA = True

    # Model
    model = Model(dropout=dropout, with_CA=with_CA, with_SA=with_SA)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)
    if args.pretrained:
        model.load_state_dict(torch.load('model/vgg_pretrained_conv_dict.pth'), strict=False)
    #for i, p in enumerate(model.parameters()):
    #    if i < 26:
    #        p.requires_grad = False

    loss_f = EdgeHoldLoss(device)

    if target_size[0] % 32 != 0 or target_size[1] % 32 != 0:
        raise ValueError('Image height and wight must be a multiple of 32')

    # data generator
    traingen = getTrainGenerator(train_path, target_size, batch_size, israndom=True)
    testgen = getValidationGenerator(test_path, target_size, batch_size, israndom=False)

    # train

    i = 0
    n = 0
    global_Fb = 0
    writer = SummaryWriter(comment='(pretrained,nofreezing,1.12,lr:1e-3 scheduler,totalalpha:0.8)')

    print('start training!')
    start_time = time.time()
    for epoch in range(epochs):
        model.train()

        # if epoch > 40:
        #     lr = lr_scheduler(epoch-40, 1e-3)
        # else:
        #     lr = lr_scheduler(epoch, 1e-2)

        # if epoch > 40:
        #     alpha_sal = 0.7
        # else:
        #     alpha_sal = 1

        alpha_sal = 0.8
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_scheduler(epoch, 1e-3), momentum=0.9)
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0004, momentum=0.9)
        mae_train_list = []
        Fb_train_list = []
        for step in range(steps_per_epoch):
            i += 1
            optimizer.zero_grad()
            imgs, masks = traingen.__next__()
            imgs.requires_grad_(True)
            masks.requires_grad_(False)
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)

            loss = loss_f(outputs, masks, alpha_sal)
            loss.backward()
            optimizer.step()

            masks = masks.view((-1))
            preds = outputs.view((-1))
            preds = preds > threshold
            preds = preds.cpu().numpy()
            masks = masks.cpu().numpy()
            TP, TN, FN, FP = 0, 0, 0, 0
            TP += ((preds == 1) & (masks == 1)).sum()
            TN += ((preds == 0) & (masks == 0)).sum()
            FN += ((preds == 0) & (masks == 1)).sum()
            FP += ((preds == 1) & (masks == 0)).sum()
            # print(TP, TN, FN, FP)
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            acc = (TP + TN) / (TP + TN + FP + FN)
            Fb = 1.3 * r * p / (r + 0.3 * p)
            ae = np.mean(np.abs(masks - preds))
            Fb_train_list.append(Fb)
            mae_train_list.append(ae)
            writer.add_scalar('Train/Loss', loss, i)
            writer.add_scalar('Train/Accuary', acc, i)

            if i % args.log_interval == 0:
                secs = time.time()-start_time
                print('TIME[%02d:%02d:%02d] EPOCH[%d/%d] STEP[%d/%d] loss: %f Fb:%f MAE: %f alpha_sal: %f' % (secs//3600, secs//60%60, secs%60, epoch+1, epochs, step+1, steps_per_epoch, loss.item(), Fb, ae, alpha_sal))

        Fb_train = np.mean(Fb_train_list)
        mae_train = np.mean(mae_train_list)
        writer.add_scalar('Train/FbScore', Fb_train, epoch)
        writer.add_scalar('Train/MAE', mae_train, epoch)

        if (epoch+1) % args.save_interval == 0:
            print('start validating!')
            model.eval()
            TP, TN, FN, FP = 0, 0, 0, 0
            mae_list = []
            Fb_list = []
            for step in range(test_steps):
                n+=1
                imgs, masks = testgen.__next__()
                imgs = imgs.to(device)
                outputs = model(imgs)
                masks=masks.to(device)
                VALloss = loss_f(outputs, masks, alpha_sal)
                masks = masks.view((-1))
                preds = outputs.view((-1))
                preds = preds > threshold
                preds = preds.cpu().numpy()
                masks = masks.cpu().numpy()
                TP += ((preds == 1) & (masks == 1)).sum()
                TN += ((preds == 0) & (masks == 0)).sum()
                FN += ((preds == 0) & (masks == 1)).sum()
                FP += ((preds == 1) & (masks == 0)).sum()
                #print(TP, TN, FN, FP)
                p = TP / (TP + FP)
                r = TP / (TP + FN)
                acc = (TP + TN) / (TP + TN + FP + FN)

                Fb = 1.3 * r * p / (r + 0.3 * p)
                Fb_list.append(Fb)

                ae = np.mean(np.abs(masks-preds))
                mae_list.append(ae)

                writer.add_scalar('Test/Loss', VALloss, n)
                writer.add_scalar('Test/Accuary', acc, n)
                if (step + 1) % args.log_interval == 0:

                    print('VAL STEP[%d/%d] precision: %.3f, recall: %.3f, Fb score: %.3f, acc: %.3f, MAE: %.3f, VALLOSS: %.3f' % (step + 1, test_steps, p, r, Fb, acc, ae, VALloss))
                    f = open('result.txt', 'a+')
                    f.writelines('EPOCH[%d] VAL STEP[%d/%d] precision: %.3f, recall: %.3f, Fb score: %.3f, acc: %.3f, MAE: %.3f, VALLOSS: %.3f' % (epoch, step + 1, test_steps, p, r, Fb, acc, ae, VALloss) + '\n')
                    f.close()
            Fb = np.mean(Fb_list)
            mae = np.mean(mae_list)
            writer.add_scalar('Test/FbScore', Fb, epoch)
            writer.add_scalar('Test/MAE', mae, epoch)
            if Fb > global_Fb:
                print('get better performance from %.3f to %.3f with MAE being %.3f), saving model...' % (global_Fb, Fb, mae))
                global_Fb = Fb
                torch.save(model.state_dict(), model_name)






