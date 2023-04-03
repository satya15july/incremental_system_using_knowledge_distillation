import numpy as np
import argparse

import torch
import torchvision
import torch.nn.functional as F
from torch.utils import tensorboard
from torch.optim import Adam
import os
import config
from network import ResNet18

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default='teacher_model.pth', help="model name")
args = vars(ap.parse_args())

niter = config.EPOCH
lr=1e-4
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
    train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)

print("len of train_loader {}".format(len(train_loader)))
print("len of val_loader {}".format(len(val_loader)))


def train():
    num_classes = config.OLD_CLASSES + config.NEW_CLASSES
    teacher_net = ResNet18(num_classes=num_classes).cuda()
    print("teacher_net: {}".format(teacher_net))

    opt = Adam(teacher_net.parameters(), lr=lr)
    for epoch in range(niter):
        teacher_net.train()
        losses, loss_Cs = [], []
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            y_pred = teacher_net(x)[:, :config.TOTAL_CLASSES - config.NEW_CLASSES]

            loss_C = F.cross_entropy(y_pred, y).mean()
            loss = loss_C
            # print("train loss: {}".format(loss))

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            loss_Cs.append(loss_C.item())

            ########################################################

        loss = np.array(losses, dtype=np.float).mean()

        print('| epoch %d, train_loss %.4f' % (epoch, loss))
        if epoch % 2 == 0:
            teacher_net.eval()
            with torch.no_grad():
                cor_num, total_num = 0, 0
                for x, y in val_loader:
                    correct_num, total = 0, 0
                    #print("x: {}".format(x))
                    x, y = x.cuda(), y.numpy()
                    y_pred = teacher_net(x).cpu().numpy()
                    #print("y_pred.shape: {}".format(y_pred.shape))
                    y_pred = y_pred.argmax(axis=-1)
                    #print("y_pred: {}".format(y_pred))
                    #print("y: {}".format(y))
                    correct_num += (y_pred == y).sum()
                    total += y.shape[0]
                    acc = correct_num / total * 100
                    cor_num += correct_num
                    total_num += total
                    print('test_acc %.4f' % (acc))
                    acc = cor_num / total_num * 100
                    print('test_acc_total %.4f' % acc)

    model_name = os.path.join(config.MODEL_PATH, args['model'])
    torch.save(teacher_net.state_dict(), model_name)

if __name__ == '__main__':
    train()
