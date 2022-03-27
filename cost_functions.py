import torch
from torch import nn
from torch import functional as F

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
    
    

    
    


    



