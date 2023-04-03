import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils import tensorboard
from torch.optim import Adam
import copy
import argparse
import os

import config
from network import ResNet18
from utils import grad_cam_loss
from config import *

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--teacher", default='teacher_model.pth', help="model name")
ap.add_argument("-s", "--student", default='student_model.pth', help="model name")
args = vars(ap.parse_args())

TEACHER_MODEL = args['teacher']
STUDENT_MODEL = args['student']

if os.path.exists(config.MODEL_PATH) == False:
    os.makedirs(config.MODEL_PATH)

RESUME = config.RESUME
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
class_indices = np.isin(cifar_dataset.targets, [cifar_dataset.class_to_idx[c] for c in config.select_classes_10_16])
cifar_dataset.targets = np.array(cifar_dataset.targets)[class_indices].tolist()
cifar_dataset.data = cifar_dataset.data[class_indices]

train_size = int(0.6 * len(cifar_dataset))
val_size = len(cifar_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(cifar_dataset, [train_size, val_size])

# Create a data loader for the subset of classes
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)

print("len of train_loader {}".format(len(train_loader)))
print("len of val_loader {}".format(len(val_loader)))

niter = config.EPOCH
lr=1e-4
beta=0.5
gamma=0.5

num_classes = config.OLD_CLASSES + config.NEW_CLASSES
teacher_net = ResNet18(num_classes=num_classes).cuda()
teacher_net.load_state_dict(torch.load(TEACHER_MODEL))
teacher_net.requires_grad_(False)
teacher_net.eval()
print("teacher_net: {}".format(teacher_net))

T = config.TEMPRATURE
old_classes = config.OLD_CLASSES
new_classes = config.NEW_CLASSES

student_net = copy.deepcopy(teacher_net)
student_net.requires_grad_(True)
if RESUME:
    student_net.load_state_dict(torch.load(STUDENT_MODEL))

teacher_net.to('cuda')
student_net.to('cuda')

opt = Adam(student_net.parameters(), lr=lr)
for epoch in range(niter):
    student_net.train()
    losses, loss_Cs, loss_Ds, loss_ADs = [], [], [], []

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        y_pred_old = teacher_net(x)[:, :config.TOTAL_CLASSES - config.NEW_CLASSES]
        y_pred_new = student_net(x)

        loss_C = F.cross_entropy(y_pred_new, y).mean()

        loss_D = F.binary_cross_entropy_with_logits(y_pred_new[:, :-new_classes], y_pred_old.detach().sigmoid())
        loss_AD = grad_cam_loss(teacher_net.feature, y_pred_old, student_net.feature, y_pred_new[:, :-config.NEW_CLASSES])
        loss = loss_C + loss_D * beta + loss_AD * gamma

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        loss_Cs.append(loss_C.item())
        loss_Ds.append(loss_D.item())
        loss_ADs.append(loss_AD.item())

        torch.cuda.empty_cache()

    loss = np.array(losses, dtype=np.float).mean()
    loss_C = np.array(loss_Cs, dtype=np.float).mean()
    loss_D = np.array(loss_Ds, dtype=np.float).mean()
    loss_AD = np.array(loss_ADs, dtype=np.float).mean()

    print('| epoch %d, train_loss %.4f, train_loss_C %.4f, train_loss_D %.4f, train_loss_AD %.4f' % (
    epoch, loss, loss_C, loss_D, loss_AD))

    if epoch % 2 == 0:
        student_net.eval()
        with torch.no_grad():
            cor_num, total_num = 0, 0
            for x, y in val_loader:
                correct_num, total = 0, 0
                x, y = x.cuda(), y.numpy()
                y_pred = student_net(x).cpu().numpy()
                y_pred = y_pred.argmax(axis=-1)
                if LOGGING:
                    print("y_pred.shape: {}".format(y_pred.shape))
                    print("y_pred: {}".format(y_pred))
                    print("y: {}".format(y))
                correct_num += (y_pred == y).sum()
                total += y.shape[0]
                acc = correct_num / total * 100
                cor_num += correct_num
                total_num += total
                print('test_acc %.4f' % (acc))
                acc = cor_num / total_num * 100
                print('test_acc_total %.4f' % acc)
    if (epoch + 1) % SAVE_INTERVAL == 0:
        MODEL_NAME = "student_model_{}.pth".format(epoch+1)
        MODEL_PATH = os.path.join(config.MODEL_PATH, MODEL_NAME)
        torch.save(student_net.state_dict(), MODEL_PATH)


torch.save(student_net.state_dict(), STUDENT_MODEL)




