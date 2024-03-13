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

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.handlelength'] = 3
plt.rcParams['legend.facecolor'] = 'inherit'

def vis_cam_on_image(img, mask, alpha=1.0):
    # mask = np.repeat(mask, 16, axis=0)  # 内挿せずに拡大
    # mask = np.repeat(mask, 16, axis=1)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap * alpha + np.float32(img)
    cam = cam / np.max(cam)
    cam = cv2.cvtColor(np.array(cam), cv2.COLOR_RGB2BGR)
    
    return cam, heatmap

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    
    return result

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

def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean, std),
    ])
    test_dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=test_transform, download=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    class_names = test_dataset.classes

    method = 'img'
    output = 'map'
    os.makedirs(method, exist_ok=True)
    os.makedirs(output, exist_ok=True)

    model = timm.create_model("vit_base_patch16_224", pretrained=True) 
    # vit_sd=torch.load(weight_name)["model"]
    # model.load_state_dict(vit_sd)
    model.to('cpu')
    model.eval()
    
    insp_names = [f'blocks.{i}.attn.attn_drop' for i in range(12)]
    insp = ModelInspector(model)
    
    cnt = 0
    
    for n, (sample, label) in enumerate(testloader):
        
        print(f'Count: {cnt}')

        sample = sample.to('cpu')
        label = label.to('cpu')
        output = model(sample)

        sample = sample.to('cpu').detach().numpy().copy()[0]
        sample = sample.transpose(1, 2, 0) * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
        
        for i in range(12):

            module_insp = insp[insp_names[i]]
            attn = module_insp.forward_inputs[0].mean(dim=1)
            attn = attn[0, 0, 1:].reshape(14, 14)  # いわゆるAttention mapにする

            attn = attn.squeeze().to('cpu').detach().numpy().copy()
            attn = min_max(attn)  # softmaxと同じ方向にmin-maxをかける場合はaxis=1

            result, _ = vis_cam_on_image(sample, attn)
            result = (result * 255).astype(np.uint8)

            plt.subplot(3, 4, i+1)
            plt.tick_params(bottom=False,
                left=False,
                right=False,
                top=False)
            plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
            plt.imshow(result)

        plt.savefig(f'{output}/attn{cnt}.png')
        plt.close()
        cnt += 1

if __name__=='__main__':
    main()