from model import Model
import numpy as np
import cv2
import os
from model import Model
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt

def padding(x):
    h,w,c = x.shape
    size = max(h,w)
    paddingh = (size-h)//2
    paddingw = (size-w)//2
    temp_x = np.zeros((size,size,c))
    temp_x[paddingh:h+paddingh,paddingw:w+paddingw,:] = x
    return temp_x

def load_image(path):
    x = cv2.imread(path)
    sh = x.shape
    x = np.array(x, dtype=np.float32)
    x = x[..., ::-1]
    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    x = padding(x)
    x = cv2.resize(x, target_size , interpolation=cv2.INTER_LINEAR)
    x = np.expand_dims(x, 0)    #(1,256,256,3)
    return x , sh



def cut(pridict,shape):
    h,w,c = shape
    size = max(h, w)
    pridict = cv2.resize(pridict, (size,size), interpolation=cv2.INTER_LINEAR)
    paddingh = (size - h) // 2
    paddingw = (size - w) // 2
    return pridict[paddingh:h + paddingh, paddingw:w + paddingw]

def laplace_edge(x):
    laplace = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge = cv2.filter2D(x/255.,-1,laplace)
    edge = np.maximum(np.tanh(edge), 0)
    edge = edge * 255
    edge = np.array(edge, dtype=np.uint8)
    return edge

def getres(pridict,shape):
    pridict = pridict*255
    pridict = np.array(pridict, dtype=np.uint8)
    pridict = np.squeeze(pridict)
    pridict = cut(pridict, shape)
    return pridict

dropout = False
with_CA = True
with_SA = True

# Model
model = Model(dropout=dropout, with_CA=with_CA, with_SA=with_SA)
new_state_dict = OrderedDict()
state_dict=torch.load('model/vgg16_saliency.pth', map_location=torch.device('cpu'))

for k, v in state_dict.items():
    name=k.replace('module.', '')
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

target_size = (256,256)
if target_size[0] % 32 != 0 or target_size[1] % 32 != 0:
    raise ValueError('Image height and wight must be a multiple of 32')

image_path='DUTS-TE-Image/ILSVRC2012_test_00007449.jpg'
GR_PATH = 'DUTS-TE-Mask/ILSVRC2012_test_00007449.png'
with torch.no_grad():
    img, shape = load_image(image_path)
    img = torch.from_numpy(img).permute(0, 3, 1, 2)
    img = img.float()
    sa = model(img)
    #sa = sa > 0.5
    sa = getres(sa, shape)
    plt.title('Saliency')
    plt.subplot(221)
    plt.imshow(cv2.imread(image_path)[..., ::-1])
    plt.subplot(222)
    edge = laplace_edge(sa)
    plt.imshow(edge, cmap='gray')

    plt.subplot(223)
    plt.imshow(sa, cmap='gray')
    plt.subplot(224)
    plt.imshow(cv2.imread(GR_PATH)[..., ::-1])
    plt.show()

