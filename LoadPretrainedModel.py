import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
vgg16 = models.vgg16(pretrained=True)
pretrained_state_dict = vgg16.state_dict()
mapping = {'conv1.weight':'features.0.weight',
'conv1.bias':'features.0.bias',
'conv2.weight':'features.2.weight',
'conv2.bias':'features.2.bias',
'conv3.weight':'features.5.weight',
'conv3.bias':'features.5.bias',
'conv4.weight':'features.7.weight',
'conv4.bias':'features.7.bias',
'conv5.weight':'features.10.weight',
'conv5.bias':'features.10.bias',
'conv6.weight':'features.12.weight',
'conv6.bias':'features.12.bias',
'conv7.weight':'features.14.weight',
'conv7.bias':'features.14.bias',
'conv8.weight':'features.17.weight',
'conv8.bias':'features.17.bias',
'conv9.weight':'features.19.weight',
'conv9.bias':'features.19.bias',
'conv10.weight':'features.21.weight',
'conv10.bias':'features.21.bias',
'conv11.weight':'features.24.weight',
'conv11.bias':'features.24.bias',
'conv12.weight':'features.26.weight',
'conv12.bias':'features.26.bias',
'conv13.weight':'features.28.weight',
'conv13.bias':'features.28.bias'}
new_state_dict = { k : pretrained_state_dict[v] for k,v in mapping.items() }
#new_model.load_state_dict(new_state_dict, strict=False)
# Save the vgg state dict for loading later
torch.save(new_state_dict, 'model/vgg_pretrained_conv_dict.pth')


# At runtime
# model.load_state_dict(torch.load('vgg_pretrained_conv_dict.pth'), strict=False)