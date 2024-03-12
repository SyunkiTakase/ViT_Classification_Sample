import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from functools import partial
from time import time
from tqdm import tqdm
import numpy as np
from PIL import Image

def train(device, train_loader, model, criterion, optimizer, lr_scheduler, scaler, use_amp, epoch):
    model.train()
    
    sum_loss = 0.0
    count = 0

    for img,label in tqdm(train_loader):
        img = img.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).long()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logit = model(img)
            loss = criterion(logit, label)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()
        count += torch.sum(logit.argmax(dim=1) == label).item()
        
    lr_scheduler.step(epoch)

    return sum_loss, count

def test(device, test_loader, model):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    sum_loss = 0.0
    count = 0

    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).long()
            
            logit = model(img)
            loss = criterion(logit, label)
            
            sum_loss += loss.item()
            count += torch.sum(logit.argmax(dim=1) == label).item()

    return sum_loss, count

def saves_train(device, train_loader, model, criterion, optimizer, lr_scheduler, scaler, use_amp, epoch):
    model.train()
    
    sum_loss = 0.0
    count = 0

    for img,label in tqdm(train_loader):
        img = img.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).long()

        for xxx in range(len(img_mixup)):
            imgs = (img[xxx].to('cpu').detach().numpy().transpose(1, 2, 0)).copy()
            img_pil = Image.fromarray((imgs*255).astype(np.uint8))
            path = './IMG/img' + str(xxx) + '.png'
            img_pil.save(path)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logit = model(img)
            loss = criterion(logit, label)

            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()
        count += torch.sum(logit.argmax(dim=1) == label).item()
        
    lr_scheduler.step(epoch)

    return sum_loss, count