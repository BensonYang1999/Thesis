import clip
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class clip_loss(nn.Module):
    def __init__(self, weight=1, consistency=False, trajectory=False):
        super(clip_loss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.weight = weight
        self.consistency = consistency
        self.trajectory = trajectory

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
        
        # pred = pred.permute(0, 1, 3, 4, 2)
        # target = target.permute(0, 1, 3, 4, 2)
        # total_loss = 0
        # for i in range(b):
        #     pred_img = self.preprocess(Image.fromarray(((pred[i][0].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
        #     target_img = self.preprocess(Image.fromarray(((target[i][0].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
        #     with torch.no_grad():
        #         pred_feat_prev = self.model.encode_image(pred_img)
        #         target_feat_prev = self.model.encode_image(target_img)
        #     for j in range(1, t):
        #         pred_img = self.preprocess(Image.fromarray(((pred[i][j].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
        #         target_img = self.preprocess(Image.fromarray(((target[i][j].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
        #         with torch.no_grad():
        #             pred_feat = self.model.encode_image(pred_img)
        #             target_feat = self.model.encode_image(target_img)

        #             loss = 1 - torch.cosine_similarity(pred_feat - pred_feat_prev, target_feat - target_feat_prev)
        #             total_loss += loss

        #             pred_feat_prev = pred_feat
        #             target_feat_prev = target_feat
        # return (total_loss * self.weight / (b * (t-1)))

        pred = pred.permute(0, 1, 3, 4, 2)
        target = target.permute(0, 1, 3, 4, 2)
        total_loss = 0
        if self.consistency:
            consistency_loss = 0
        else:
            consistency_loss = None
        traj_a, traj_b = [], []
        if self.trajectory:
            trajectory_loss = 0
        else:
            trajectory_loss = None
        
        for i in range(b):
            pred_img = self.preprocess(Image.fromarray(((pred[i][0].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
            target_img = self.preprocess(Image.fromarray(((target[i][0].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_feat_prev = self.model.encode_image(pred_img)
                target_feat_prev = self.model.encode_image(target_img)
                traj_a.append(pred_feat_prev[0])
                traj_b.append(target_feat_prev[0])
                total_loss += 1 - torch.cosine_similarity(pred_feat_prev, target_feat_prev)
            for j in range(1, t):
                pred_img = self.preprocess(Image.fromarray(((pred[i][j].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
                target_img = self.preprocess(Image.fromarray(((target[i][j].cpu().detach().numpy()+1)/2*255).astype(np.uint8))).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred_feat = self.model.encode_image(pred_img)
                    target_feat = self.model.encode_image(target_img)
                    traj_a.append(pred_feat[0])
                    traj_b.append(target_feat[0])

                    if self.consistency:
                        consistency_loss += 1 - torch.cosine_similarity(pred_feat - pred_feat_prev, target_feat - target_feat_prev)
                    total_loss += 1 - torch.cosine_similarity(pred_feat_prev, target_feat_prev)

                    pred_feat_prev = pred_feat
                    target_feat_prev = target_feat
            
            if self.trajectory:
                traj_a = torch.stack(traj_a, dim=0)
                traj_b = torch.stack(traj_b, dim=0)
                trajectory_loss += self.hausdorff_distance(traj_a, traj_b)
                traj_a, traj_b = [], []
        
        total_loss = total_loss * self.weight / (b * t)
        if self.consistency:
            consistency_loss = consistency_loss * (self.weight / 5) / (b * (t-1))
        if self.trajectory:
            trajectory_loss = trajectory_loss * (0.01) / b

        return total_loss, consistency_loss, trajectory_loss
        # return None, consistency_loss, trajectory_loss
        # return (total_loss * self.weight / (b * t)), (consistency_loss * (self.weight / 5) / (b * (t-1))), (trajectory_loss * (0.01) / b)
        # return None, None, (trajectory_loss * (0.03) / b)
        # return None, (consistency_loss * (self.weight / 10) / (b * (t-1)))
    
    def hausdorff_distance(self, x, y):
        N, D = x.shape
        
        x = x.unsqueeze(2)  # (N, 1, D)
        y = y.unsqueeze(1)  # (1, N, D)
        diff = x - y  # (N, N, D)
        dist = torch.norm(diff, dim=-1)  # (N, N)

        # Compute the directed Hausdorff distance from x to y
        max_dist_x_to_y, _ = torch.max(torch.min(dist, dim=1)[0], dim=0)  # ()
        # Compute the directed Hausdorff distance from y to x
        max_dist_y_to_x, _ = torch.max(torch.min(dist, dim=0)[0], dim=0)  # ()
        
        # The Hausdorff distance is the maximum of the two directed distances
        hausdorff_dist = torch.max(max_dist_x_to_y, max_dist_y_to_x)
        
        return hausdorff_dist
