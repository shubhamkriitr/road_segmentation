import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.functional
from commonutil import resolve_device


base_bce = torch.nn.BCELoss(reduction='none')

def weighted_bce_loss(y_pred, y_true):
    y_true = y_true[:, :1]
    int_loss = base_bce(y_pred, y_true)
    pred_round = y_pred.detach().round()
    weights = torch.where((pred_round==0) & (y_true==1), 5, 1)
    return torch.mean(weights*int_loss)

class Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Loss, self).__init__()
        if size_average is not None or reduce is not None:
            raise NotImplementedError()
        self.reduction = reduction


class WeightedLoss(Loss):
    def __init__(self, weight = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight) # self.weight will be set
        self.weight
        

class GeneralizeDiceLoss(WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)
    
    def forward(self, input_, target):
        """Assumes input_ and target are of shape: (Batch, Channel, H, W)
        Where each slice along `$i^{th}$` channel is a probabilty map for the
        output class with the id `$i$`.
        """
        raise NotImplementedError()

class BinaryGeneralizeDiceLoss(Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input_, target):
        """Assumes input_ and target are of shape: (Batch, 1, H, W)
        Where for each item in the batch, the slice along channel 
        is a probabilty map for ouput class with label id `1`.
        """
        cost = 1 - 2*torch.sum(input_*target)/(torch.sum(input_+target) + 1e-7)
        return cost


class EdgeWeightedBinaryGeneralizeDiceLoss(Loss):
    def __init__(self, edge_weight_factor=10, size_average=None,
                 reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.edge_layer = EdgeWeightingKernel(
            edge_weight_factor=edge_weight_factor)
        

    def forward(self, input_, target):
        """Assumes input_ and target are of shape: (Batch, 1, H, W)
        Where for each item in the batch, the slice along channel 
        is a probabilty map for ouput class with label id `1`.
        """
        edge_weights = self.edge_layer(target)
        cost = 1 - 2*torch.sum(input_*target*edge_weights)\
                 /  (torch.sum(edge_weights*(input_+target)) + 1e-7)
        return cost
class BinaryGeneralizeDiceLossV2(BinaryGeneralizeDiceLoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, input_, target):
        cost =  super().forward(input_, target)
        cost += super().forward(1 - input_, 1 - target)
        return cost/2.0

class ClassWeightedBinaryGeneralizeDiceLoss(WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)
    
    def forward(self, input_, target):
        cost = 1.0 - self.weight[0]*torch.sum(input_*target)/(torch.sum(input_+target) + 1e-7)
        input_ = 1.0 - input_
        target = 1.0 - target
        cost += - self.weight[0]*torch.sum(input_*target)/(torch.sum(input_+target) + 1e-7)
        return cost


# Fixed-Kernel Modules
class EdgeWeightingKernel(nn.Module):
    def __init__(self, edge_weight_factor=10, num_channels=1, device=resolve_device()) -> None:
        """
        `edge_weight_factor` controls the weight given to edges as follows:
        `final_weight` = `normalized_edge_map`*`edge_weight_factor` + 1.0
        """
        super().__init__()
        self.num_channels = num_channels
        self.device = device
        self.edge_weight_factor = edge_weight_factor
        self._init_edge_kernel()
        
    def forward(self, img):
        """
        Applies a 3 x 3 edge detection kernel
        to `img`. `img` must have shape like: (Batch, num_channels, H, W).
        """
        with torch.no_grad():
            edge_map =  F.conv2d(img, self.edge_kernel, padding="same")
            edge_map = torch.abs(edge_map)
            max_values = torch.amax(edge_map, dim=(2, 3), keepdim=True) # along 
            edge_map = edge_map/max_values
        
        return edge_map*self.edge_weight_factor + 1.0 # +1.0 for base map
        
    
    def _init_edge_kernel(self):
        kernel = torch.tensor(
            data=[[1,  2,  -1],
                  [2,  0, -2],
                  [1, -2, -1]],
            dtype=torch.float32, #TODO: take dtype from some central module/config
            device=self.device,
            requires_grad=False
        )
        self.edge_kernel = kernel.view(1, 1, 3, 3).repeat(1, self.num_channels, 1, 1)
        self.edge_kernel.requires_grad = False
        

if __name__ == "__main__":
    

    def test_1(loss_func, input_, target):
        print("Input:\n", input_)
        print("target:\n", target)
        gdl = loss_func(input_, target)
        print(f"gdl {gdl}")

    def run_tests(loss_func):
        print("="*20)
        print(loss_func.__class__.__name__)
        print("="*20)
        p = torch.tensor(
            [
                [[1, 0],
                [0, 0]],

                [[1, 1],
                [1, 1]]
            ],
            dtype=torch.float32
        )

        g = torch.tensor(
            [
                [[1, 0],
                [0, 0]],

                [[1, 1],
                [1, 1]]
            ],
            dtype=torch.float32
        )

        test_1(loss_func, p, g)
        g = 1 - g
        test_1(loss_func, p, g)
        g[0, 0, 0] = 1
        test_1(loss_func, p, g)
    
    run_tests(BinaryGeneralizeDiceLoss())
    run_tests(BinaryGeneralizeDiceLossV2())
    run_tests(ClassWeightedBinaryGeneralizeDiceLoss(
                weight=torch.tensor([1.0, 0.0])))
    run_tests(ClassWeightedBinaryGeneralizeDiceLoss(
                weight=torch.tensor([1.0, 1.0])))
    run_tests(ClassWeightedBinaryGeneralizeDiceLoss(
                weight=torch.tensor([0.5, 0.5])))
    
    

    
    


    



