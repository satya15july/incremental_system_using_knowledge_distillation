import torch

import config
from network import ResNet18
import numpy as np
import argparse
import os

import torch
import torchvision

from torch.utils import tensorboard


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--teacher", default='teacher_model.pth', help="model name")
ap.add_argument("-s", "--student", default='student_model.pth', help="model name")
args = vars(ap.parse_args())

TEACHER_MODEL = args['teacher']
STUDENT_MODEL = args['student']

from config import *
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

cifar_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True, transform=transform, download=True
)

# Select a subset of classes
class_indices = np.isin(cifar_dataset.targets, [cifar_dataset.class_to_idx[c] for c in config.select_classes_1_10])
cifar_dataset.targets = np.array(cifar_dataset.targets)[class_indices].tolist()
cifar_dataset.data = cifar_dataset.data[class_indices]

train_size = int(0.8 * len(cifar_dataset))
val_size = len(cifar_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(cifar_dataset, [train_size, val_size])

# Create a data loader for the subset of classes
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

print("len of train_loader {}".format(len(train_loader)))
print("len of test_loader {}".format(len(val_loader)))
old_classes = config.OLD_CLASSES
new_classes = config.NEW_CLASSES
total_classes = config.TOTAL_CLASSES

teacher_net = ResNet18(num_classes=total_classes).to('cpu')
teacher_net.load_state_dict(torch.load(TEACHER_MODEL))
print("TeacherNet: {}".format(teacher_net))


student_net = ResNet18(num_classes=total_classes).to('cpu')
print("StudentNet: {}".format(student_net))
student_net.load_state_dict(torch.load(STUDENT_MODEL))


print("StudentNet: {}".format(student_net))

teacher_net.eval()
student_net.eval()
with torch.no_grad():
    cor_num_t, cor_num_s, total_num = 0, 0, 0
    for x, y in train_loader:
        correct_num_t, correct_num_s, total = 0, 0, 0
        # print("x: {}".format(x))
        x, y = x, y.numpy()
        y_pred_t = teacher_net(x)[:, :old_classes].cpu().numpy()
        y_pred_s = student_net(x)[:, :old_classes].cpu().numpy()
        print("y_pred_t.shape: {}".format(y_pred_t.shape))
        print("y_pred_s.shape: {}".format(y_pred_s.shape))

        y_pred_t = y_pred_t.argmax(axis=-1)
        y_pred_s = y_pred_s.argmax(axis=-1)
        print("y: {}".format(y))
        print("y_pred_t: {}".format(y_pred_t))
        print("y_pred_s: {}".format(y_pred_s))

        # print("y_pred: {}".format(y_pred))
        # print("y: {}".format(y))
        correct_num_t += (y_pred_t == y).sum()
        correct_num_s += (y_pred_s == y).sum()
        total += y.shape[0]
        acc_t = correct_num_t / total * 100
        acc_s = correct_num_s / total * 100
        cor_num_t += correct_num_t
        cor_num_s += correct_num_s
        total_num += total
        #print('Teacher test_acc %.4f' % (acc_t))
        #print('Student test_acc %.4f' % (acc_s))
        acc_t = cor_num_t / total_num * 100
        acc_s = cor_num_s / total_num * 100
        print('Teacher test_acc_total %.4f' % acc_t)
        print('Student test_acc_total %.4f' % acc_s)


