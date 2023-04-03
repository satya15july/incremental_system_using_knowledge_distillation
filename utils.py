import torch
import torch.nn.functional as F
from config import LOGGING
def grad_cam_loss(feature_o, out_o, feature_n, out_n):
    if LOGGING:
        print("grad_cam_loss---->")
        print("feature_o: {}, out_o: {}".format(feature_o.shape, out_o.shape))
        print("feature_n: {}, out_n: {}".format(feature_n.shape, out_n.shape))
        #print("feature_o: {}, out_o: {}, feature_n: {}".format(feature_o, out_o, feature_n, out_n))

    batch = out_n.size()[0]
    index = out_n.argmax(dim=-1).view(-1, 1)
    onehot = torch.zeros_like(out_n)
    onehot.scatter_(-1, index, 1.)
    out_o, out_n = torch.sum(onehot * out_o), torch.sum(onehot * out_n)

    grads_o = torch.autograd.grad(out_o, feature_o)[0]
    grads_n = torch.autograd.grad(out_n, feature_n, create_graph=True)[0]

    weight_o = grads_o.mean(dim=(2, 3)).view(batch, -1, 1, 1)
    weight_n = grads_n.mean(dim=(2, 3)).view(batch, -1, 1, 1)

    cam_o = F.relu((grads_o * weight_o).sum(dim=1))
    cam_n = F.relu((grads_n * weight_n).sum(dim=1))

    # normalization
    cam_o = F.normalize(cam_o.view(batch, -1), p=2, dim=-1)
    cam_n = F.normalize(cam_n.view(batch, -1), p=2, dim=-1)

    loss_AD = (cam_o - cam_n).norm(p=1, dim=1).mean()

    if LOGGING:
        print("loss_AD: {}".format(loss_AD))

    return loss_AD

