import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import cv2
import argparse
import os
import config
from network import ResNet18

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--teacher", default='teacher_model.pth', help="model name")
ap.add_argument("-s", "--student", default='student_model.pth', help="model name")
args = vars(ap.parse_args())

TEACHER_MODEL = os.path.join(config.MODEL_PATH, args['teacher'])
STUDENT_MODEL = os.path.join(config.MODEL_PATH, args['student'])


teacher_net = ResNet18(num_classes=config.TOTAL_CLASSES).to('cpu')
teacher_net.load_state_dict(torch.load(TEACHER_MODEL))
print("TeacherNet: {}".format(teacher_net))

teacher_net.requires_grad_(False)
teacher_net.eval()


student_net = ResNet18(num_classes=config.TOTAL_CLASSES).to('cpu')
student_net.load_state_dict(torch.load(STUDENT_MODEL))
student_net.requires_grad_(False)
student_net.eval()


image_path = 'test_image/apple_3.jpeg'
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (32, 32))
rgb_img = np.float32(rgb_img) / 255

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

teacher_net.eval()
#student_net.eval()
with torch.no_grad():
    x = transform(rgb_img).unsqueeze(0)
    gs_t = teacher_net(x)
    gs_f_t = teacher_net.feature
    print("Shape teacher_net.feature.shape: {}".format(teacher_net.feature.shape))
    print("Shape gs_t.shape: {}".format(gs_t.shape))
    gs_f_t = gs_f_t.pow(2).mean(1)
    print("Shape gs_t.shape: {}".format(gs_t.shape))
    #gs_s = student_net(x)

    for i in range(1,10):
        print(gs_f_t.shape)
        plt.imshow(gs_f_t[0], interpolation='bicubic')
        plt.title('Featuremap Teacher')
        plt.show()

