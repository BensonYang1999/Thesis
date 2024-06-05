import clip
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class clip_loss(nn.Module):
    def __init__(self, weight=1):
        super(clip_loss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.weight = weight

    def forward(self, pred, target):
        b, t, c, h, w = pred.shape
        # pred = pred.view(b * t , c, h, w).permute(0, 2, 3, 1)
        # target = target.view(b * t , c, h, w).permute(0, 2, 3, 1)
        # total_loss = 0
        # for i in range(b * t):
        #     pred_img = self.preprocess(Image.fromarray(((pred[i].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
        #     target_img = self.preprocess(Image.fromarray(((target[i].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
        #     with torch.no_grad():
        #         pred_feat = self.model.encode_image(pred_img)
        #         target_feat = self.model.encode_image(target_img)

        #         loss = 1 - torch.cosine_similarity(pred_feat, target_feat)
        #         total_loss += loss
        # return (total_loss * self.weight / (b * t))
        
        pred = pred.permute(0, 1, 3, 4, 2)
        target = target.permute(0, 1, 3, 4, 2)
        total_loss = 0
        for i in range(b):
            pred_img = self.preprocess(Image.fromarray(((pred[i][0].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
            target_img = self.preprocess(Image.fromarray(((target[i][0].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_feat_prev = self.model.encode_image(pred_img)
                target_feat_prev = self.model.encode_image(target_img)
            for j in range(1, t):
                pred_img = self.preprocess(Image.fromarray(((pred[i][j].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
                target_img = self.preprocess(Image.fromarray(((target[i][j].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred_feat = self.model.encode_image(pred_img)
                    target_feat = self.model.encode_image(target_img)

                    loss = 1 - torch.cosine_similarity(pred_feat - pred_feat_prev, target_feat - target_feat_prev)
                    total_loss += loss

                    pred_feat_prev = pred_feat
                    target_feat_prev = target_feat
        return (total_loss * self.weight / (b * (t-1)))
