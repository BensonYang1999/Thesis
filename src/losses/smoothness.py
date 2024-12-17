import torch
import torch.nn as nn

class smooth_loss(nn.Module):
    def __init__(self, weight=1):
        super(smooth_loss, self).__init__()
        self.weight = weight

    def forward(self, pred):
        b, t, c, h, w = pred.shape

        # grad_x = torch.abs(pred[:, :, :, :, :-1] - pred[:, :, :, :, 1:])
        # grad_y = torch.abs(pred[:, :, :, :-1, :] - pred[:, :, :, 1:, :])

        grad_t = torch.abs(pred[:, :-1, :, :, :] - pred[:, 1:, :, :, :])

        # loss = torch.mean(grad_x) + torch.mean(grad_y)
        loss = torch.mean(grad_t)

        return loss * self.weight
        
    
