import timm
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os
import random
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt

from collections import OrderedDict


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def vis_cam_on_image(img, mask, alpha=1.0):
    # mask = np.repeat(mask, 16, axis=0)  # 内挿せずに拡大
    # mask = np.repeat(mask, 16, axis=1)
    sample=img
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap * alpha + np.float32(img)
    cam = cam / np.max(cam)
    cam = cv2.cvtColor(np.array(cam), cv2.COLOR_RGB2BGR)
    return cam, heatmap

class ModuleInspector():
    def __init__(self, module):
        self.module = module
        self.forward_inputs = None    # tuple
        self.forward_output = None    # tensor
        self.backward_inputs = None   # tuple
        self.backward_outputs = None  # tuple
        self.forward_hook_handler = module.register_forward_hook(self.on_forward)
        self.backward_hook_handler = module.register_full_backward_hook(self.on_backward)
        return

    def on_forward(self, module, inputs, output):
        self.forward_inputs = [tensor if tensor is not None else tensor for tensor in inputs]
        self.forward_output = output
        return

    def on_backward(self, module, grad_inputs, grad_outputs):
        self.backward_inputs = [tensor if tensor is not None else tensor for tensor in grad_outputs]
        self.backward_outputs = [tensor if tensor is not None else tensor for tensor in grad_inputs]
        return

class ModelInspector(OrderedDict):
    def __init__(self, model):
        for module_name, module in model.named_modules():
            self[module_name] = ModuleInspector(module)
        return

testdir = ''
test_dataset = torchvision.datasets.ImageFolder(
            testdir,
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ]))
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

model = create_model(
    'cait_s24_224',
    pretrained=False,
    num_classes=1000,
    drop_block_rate=None
)
vit_sd=torch.load(weight_name)["model"]
model.load_state_dict(vit_sd)
model.to('cuda')
model.eval()
insp_names = ['blocks_token_only.0.attn.attn_drop']
insp = ModelInspector(model)

cnt=0
for n, (sample, cls) in enumerate(testloader):
    print('aaa')
    with torch.cuda.amp.autocast(enabled=True):
        sample = sample.to('cuda')
        cls = cls.to('cuda')
        output = model(sample)
    for i in range(1):
        module_insp = insp[insp_names[i]]

        attn = module_insp.forward_inputs[0].mean(dim=1)[0, 0, 1:]
        attn = attn.reshape(14, 14)  # いわゆるAttention mapにする
        sample=sample.permute(0,2,3,1)
        sample = sample.squeeze().to('cpu').detach().numpy().copy()
        attn = attn.squeeze().to('cpu').detach().numpy().copy() # Attention Weight
        attn = min_max(attn,axis=1)  # softmaxと同じ方向にmin-maxをかける場合はaxis=1
        result,heatmap = vis_cam_on_image(sample, attn)
        result = (result * 255).astype(np.uint8)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.imshow(sample)
        plt.savefig(f'./img{cnt}.png',bbox_inches="tight")
        plt.close()
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.imshow(result)
        plt.savefig(f'./attn{cnt}.png',bbox_inches="tight")
        plt.close()
    cnt+=1
