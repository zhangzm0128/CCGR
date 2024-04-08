import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, loss_name, device):
        super(Loss, self).__init__()
        self.device = device
        self.loss = getattr(self, loss_name)

    def MSE(self):
        return nn.MSELoss(reduce=True, reduction='mean')
        # return nn.MSELoss(reduce=False)
        
    def Bmode_MSE(self):
        return BmodeMSE()
    
    def MGE(self):
        return MGELoss(self.device)
        
    def MSEPenalty(self):
        return MSE_Penalty(self.device)
        
    def BmodeMSE_PLus_MSE(self):
        return BmodeMSE_and_NormalMSE()
        

class BmodeMSE(nn.Module):
    def __init__(self):
        super(BmodeMSE, self).__init__()
        self.epslion = 1e-10
        self.mse = nn.MSELoss(reduce=True, reduction='mean')
    def forward(self, x, y):
    
        # x_bmode = torch.flatten(x + self.epslion, 1, 2)
        # y_bmode = torch.flatten(y + self.epslion, 1, 2)
        
        x_bmode = x.view(x.size(0), -1) + self.epslion
        y_bmode = y.view(y.size(0), -1) + self.epslion
        
        x_bmode = torch.log10(x_bmode / torch.max(x_bmode, dim=1)[0].view(-1, 1))
        y_bmode = torch.log10(y_bmode / torch.max(y_bmode, dim=1)[0].view(-1, 1))
        
        return self.mse(x_bmode, y_bmode)
        
        
class BmodeMSE_and_NormalMSE(nn.Module):
    def __init__(self):
        super(BmodeMSE_and_NormalMSE, self).__init__()
        self.epslion = 1e-10
        self.Normal_MSE = nn.MSELoss(reduce=True, reduction='mean')
        self.Bmode_MSE = BmodeMSE()
    def forward(self, x, y):
        
        return self.Normal_MSE(x, y) + self.Bmode_MSE(x, y)
        
        
class MGELoss(nn.Module):
    def __init__(self, device):
        super(MGELoss, self).__init__()
        self.epslion = 1e-10
        self.gx = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).to(device)
        self.gy = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(device)
        self.mse = nn.MSELoss(reduce=True, reduction='mean')
        
    def get_grad(self, data):
        grad_x = F.conv2d(data, self.gx, stride=1, bias=None, padding=1)
        grad_y = F.conv2d(data, self.gy, stride=1, bias=None, padding=1)
        return torch.sqrt(grad_x.squeeze(1) ** 2 + grad_y.squeeze(1) ** 2)
        
    def forward(self, output, target):
        output_grad = self.get_grad(output.unsqueeze(1))
        target_grad = self.get_grad(target.unsqueeze(1))
        
        return self.mse(output_grad, target_grad) + self.mse(output, target)
        
class MSE_Penalty(nn.Module):
    def __init__(self, device):
        super(MSE_Penalty, self).__init__()
        self.epslion = 1e-10
        self.mse = nn.MSELoss(reduce=False)
        self.device = device
        
        self.w_scale = 16
        self.h_scale = 16
        
        self.FWHM_threshold = 0.5
        
        
    def get_penalty_weight(self, data):
        b, h, w = data.size()
        max_pos = torch.max(torch.flatten(data, 1, 2), dim=1)[1]
        penalty_weight = torch.ones_like(data)
        for x in range(b):
            point_h = max_pos[x] // w
            point_w = max_pos[x] % w
            
            if data[x, point_h, point_w] < self.FWHM_threshold:
                continue
            
            area_top = point_h - h // self.h_scale if point_h - h // self.h_scale > 0 else 0
            area_bot = point_h + h // self.h_scale if point_h + h // self.h_scale < h else h
            
            area_left  = point_w - w // self.w_scale if point_w - w // self.w_scale > 0 else 0
            area_right = point_w + w // self.w_scale if point_w + w // self.w_scale < w else w
            
            penalty_h = torch.linspace(1, 10, (area_bot - area_top + 1) // 2, device=self.device)
            penalty_w = torch.linspace(1, 10, (area_right - area_left + 1) // 2, device=self.device)
            
            if (area_bot - area_top) % 2 == 0:
                penalty_h = torch.cat([penalty_h, torch.flip(penalty_h, dims=[0])])
            else:
                penalty_h = torch.cat([penalty_h, torch.flip(penalty_h, dims=[0])[1:]])
            if (area_right - area_left) % 2 == 0:
                penalty_w = torch.cat([penalty_w, torch.flip(penalty_w, dims=[0])])
            else:
                penalty_w = torch.cat([penalty_w, torch.flip(penalty_w, dims=[0])[1:]])
            
            penalty_weight[x, area_top: area_bot, area_left: area_right] = torch.mm(penalty_h.view(-1, 1), penalty_w.view(1, -1))
        
        return penalty_weight
        
    def forward(self, output, target):
        penalty_weight = self.get_penalty_weight(target)
        
        return torch.mean(self.mse(output, target) * penalty_weight)
        
        
        
        
        
        
        
        

