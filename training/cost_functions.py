import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.functional
from utils.commonutil import resolve_device, BaseFactory

base_bce = torch.nn.BCELoss(reduction='none')


def weighted_bce_loss(input, target):
    target = target[:, :1]
    int_loss = base_bce(input, target)
    pred_round = input.detach().round()
    weights = torch.where((pred_round == 0) & (target == 1), 5, 1)
    return torch.mean(weights * int_loss)


class Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Loss, self).__init__()
        if size_average is not None or reduce is not None:
            raise NotImplementedError()
        self.reduction = reduction


class BinaryCrossEntropyLoss(Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(BinaryCrossEntropyLoss, self).__init__(size_average, reduce, reduction)
        self.bce = torch.nn.BCELoss(reduction=reduction)

    def forward(self, input, target):
        return self.bce(input.flatten(), target.flatten())


class WeightedLoss(Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)  # self.weight will be set
        self.weight


class GeneralizeDiceLoss(WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input, target):
        """Assumes input and target are of shape: (Batch, Channel, H, W)
        Where each slice along `$i^{th}$` channel is a probabilty map for the
        output class with the id `$i$`.
        """
        raise NotImplementedError()


class BinaryGeneralizeDiceLoss(Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        """Assumes input and target are of shape: (Batch, 1, H, W)
        Where for each item in the batch, the slice along channel
        is a probabilty map for ouput class with label id `1`.
        """
        cost = 1 - 2 * torch.sum(input * target) / (torch.sum(input + target) + 1e-7)
        return cost

class DiceLoss(Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        """Assumes input and target are of shape: (Batch, 1, H, W)
        Where for each item in the batch, the slice along channel
        is a probabilty map for ouput class with label id `1`.
        """
        cost = 1 - 2 * torch.sum(input * target) \
            / (torch.sum(input*input + target*target) + 1e-8)
        return cost

class PatchedBinaryGeneralizeDiceLoss(Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', patch_size=16) -> None:
        super().__init__(size_average, reduce, reduction)
        self.pooling = torch.nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, input, target):
        """Assumes input and target are of shape: (Batch, 1, H, W)
        Where for each item in the batch, the slice along channel
        is a probabilty map for ouput class with label id `1`.
        """
        # Apply mean pooling to input and target using the patch size used for evaluation
        patched_input = self.pooling(input)
        patched_target = self.pooling(target).round()
        cost = 1 - 2*torch.sum(patched_input*patched_target)/(torch.sum(patched_input+patched_target) + 1e-7)
        return cost

class ThresholdedBinaryGeneralizedDiceLoss(BinaryGeneralizeDiceLoss):
    def __init__(self, threshold=0.3, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.threshold = threshold

    def forward(self, input, target):
        input = torch.where(input > self.threshold, 1., 0.)
        return super().forward(input, target)


class EdgeWeightedBinaryGeneralizeDiceLoss(Loss):
    def __init__(self, edge_weight_factor=10, size_average=None,
                 reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.edge_layer = EdgeWeightingKernel(
            edge_weight_factor=edge_weight_factor)

    def forward(self, input, target):
        """Assumes input and target are of shape: (Batch, 1, H, W)
        Where for each item in the batch, the slice along channel
        is a probabilty map for ouput class with label id `1`.
        """
        edge_weights = self.edge_layer(target)
        cost = 1 - 2 * torch.sum(input * target * edge_weights) \
               / (torch.sum(edge_weights * (input + target)) + 1e-7)
        return cost


class BinaryGeneralizeDiceLossV2(BinaryGeneralizeDiceLoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        cost = super().forward(input, target)
        cost += super().forward(1 - input, 1 - target)
        return cost / 2.0


class ClassWeightedBinaryGeneralizeDiceLoss(WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input, target):
        cost = 1.0 - self.weight[0] * torch.sum(input * target) / (torch.sum(input + target) + 1e-7)
        input = 1.0 - input
        target = 1.0 - target
        cost += - self.weight[0] * torch.sum(input * target) / (torch.sum(input + target) + 1e-7)
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
            edge_map = F.conv2d(img, self.edge_kernel, padding="same")
            edge_map = torch.abs(edge_map)
            max_values = torch.amax(edge_map, dim=(2, 3), keepdim=True)  # along
            edge_map = edge_map / max_values

        return edge_map * self.edge_weight_factor + 1.0  # +1.0 for base map

    def _init_edge_kernel(self):
        kernel = torch.tensor(
            data=[[1, 2, -1],
                  [2, 0, -2],
                  [1, -2, -1]],
            dtype=torch.float32,  # TODO: take dtype from some central module/config
            device=self.device,
            requires_grad=False
        )
        self.edge_kernel = kernel.view(1, 1, 3, 3).repeat(1, self.num_channels, 1, 1)
        self.edge_kernel.requires_grad = False


class SoftBootstrappedDiceLoss(Loss):
    def __init__(self, beta=0.9, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.beta = beta

    def forward(self, input, target):
        comb = self.beta * target + (1 - self.beta) * input
        cost = 1 - (1 + 2 * torch.sum(input * comb)) \
               / (1 + torch.sum(input * input + target * target))
        return cost

class TverskyLoss(Loss):
    def __init__(self, beta=0.9, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.beta = beta
        self.eps = 1e-9
        if reduction != "mean":
            raise ValueError(f"Reduction {reduction} is not allowed")

    def forward(self, input, target):
        nr = target*input
        dr = nr + self.beta*(1-target)*input + (1-self.beta)*target*(1-input)
        cost =  1. - (1 + nr)/(1 + dr + self.eps)
        return torch.mean(cost)
    
class FocalTverskyLoss(Loss):
    def __init__(self, beta=0.9, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.beta = beta
        self.eps = 1e-9
        if reduction != "mean":
            raise ValueError(f"Reduction {reduction} is not allowed")
        self._upper_cutoff = 0.3
        self._lower_cutoff = 0.05
        
        self._max_ratio = 0.0 # of positive pixels in ground truth
        self._min_ratio = 1.0

    def forward(self, input, target):
        nr = target*input
        dr = nr + self.beta*(1-target)*input + (1-self.beta)*target*(1-input)
        
        nr = torch.sum(nr, dim=(1, 2, 3), keepdim=True)
        dr = torch.sum(dr, dim=(1, 2, 3), keepdim=True)
        
        tversky_indices = nr/ (dr + self.eps)
        cost =  1. - tversky_indices
        
        gammas = self.compute_gamma(input=input, target=target)
        
        cost = torch.pow(cost, gammas)
        
        cost = torch.mean(cost)
        
        return cost
    
    def compute_gamma(self, input, target):
        positive_fractions = torch.mean(target, dim=(1, 2, 3), keepdim=True)
        
        positive_fractions = torch.clamp(positive_fractions,
                                         min=self._lower_cutoff,
                                         max=self._upper_cutoff)
        
        max_ = torch.max(positive_fractions)
        min_ = torch.min(positive_fractions)
    
        
        if min_ < self._min_ratio:
            self._min_ratio = min_
        if max_ > self._max_ratio:
            self._max_ratio = max_
        
        gammas = 3*(positive_fractions - self._min_ratio)\
                    /(self._max_ratio - self._min_ratio + 1e-9)
                    
        return gammas
        

class EdgeWeightedSoftBootstrappedDiceLoss(Loss):
    def __init__(self, beta=0.9, edge_weight_factor=10, size_average=None,
                 reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.edge_layer = EdgeWeightingKernel(
            edge_weight_factor=edge_weight_factor)
        self.beta = beta

    def forward(self, input, target):
        """Assumes input and target are of shape: (Batch, 1, H, W)
        Where for each item in the batch, the slice along channel
        is a probabilty map for ouput class with label id `1`.
        """
        edge_weights = self.edge_layer(target)
        comb = self.beta * target + (1 - self.beta) * input
        cost = 1 - (1 + 2 * torch.sum(edge_weights * input * comb)) \
               / (1 + torch.sum(edge_weights * (input * input + target * target)))
        return cost


COST_FUNCTION_NAME_TO_CLASS_MAP = {
    "ClassWeightedBinaryGeneralizeDiceLoss": \
        ClassWeightedBinaryGeneralizeDiceLoss,
    "BinaryGeneralizeDiceLoss": BinaryGeneralizeDiceLoss,
    "EdgeWeightedBinaryGeneralizeDiceLoss": EdgeWeightedBinaryGeneralizeDiceLoss,
    "weighted_bce_loss": lambda: weighted_bce_loss,
    "ThresholdedBinaryGeneralizedDiceLoss": \
        ThresholdedBinaryGeneralizedDiceLoss,
    "SoftBootstrappedDiceLoss": SoftBootstrappedDiceLoss,
    "EdgeWeightedSoftBootstrappedDiceLoss": \
        EdgeWeightedSoftBootstrappedDiceLoss,
    "PatchedBinaryGeneralizeDiceLoss": PatchedBinaryGeneralizeDiceLoss,
    "TverskyLoss": TverskyLoss,
    "DiceLoss": DiceLoss,
    "BinaryGeneralizeDiceLossV2": BinaryGeneralizeDiceLossV2,
    "FocalTverskyLoss": FocalTverskyLoss,
    "BCE": BinaryCrossEntropyLoss
}


class CostFunctionFactory(BaseFactory):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.resource_map = COST_FUNCTION_NAME_TO_CLASS_MAP

    def get(self, cost_function_class_name, config=None,
            args_to_pass=[], kwargs_to_pass={}):
        return super().get(cost_function_class_name, config,
                           args_to_pass, kwargs_to_pass)


if __name__ == "__main__":
    cf = CostFunctionFactory().get("ClassWeightedBinaryGeneralizeDiceLoss")


    def test_1(loss_func, input, target):
        print("Input:\n", input)
        print("target:\n", target)
        gdl = loss_func(input, target)
        print(f"gdl {gdl}")


    def run_tests(loss_func):
        print("=" * 20)
        print(loss_func.__class__.__name__)
        print("=" * 20)
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











