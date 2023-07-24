import torch 

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, true):
    # logcosh loss  双曲余弦的对数损失
        loss = torch.log(torch.cosh(true - pred))
        return torch.mean(loss)
    
class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
    def forward(self, pred, true):
    # 分位数损失
        assert not true.requires_grad
        assert pred.size(0) == true.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = true - pred[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
    
class mse_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, true):
    # 均方误差
        loss = torch.mean(torch.abs(pred - true)**2)
        return loss
    
class rmse_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, true):
    # 均方根误差
        # pred[pred>125] = pred[pred>125] + 0.1*(pred[pred>125]-125)**2
        loss = torch.sqrt(torch.mean(torch.abs(pred - true)**2))
        return loss

      

    



