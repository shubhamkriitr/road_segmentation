from vision_transformer.dino import vision_transformer as vits
from vision_transformer.dino.eval_linear import LinearClassifier
from vision_transformer.dino import utils
import torch
from torchvision import transforms as pth_transforms
import sys


def get_dino(model_name='vit_small', patch_size=16, n_last_blocks=4, avgpool_patchtokens=False, device='cuda',classifier = True, pretrained_classifier = True, num_labels=1000):
    if model_name in vits.__dict__.keys():
        model = vits.__dict__[model_name](patch_size=patch_size, num_classes=0)
        embed_dim = model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
        utils.load_pretrained_weights(model, "", "", model_name, patch_size)
    else:
        print(f"Unknow architecture: {model_name}")
        sys.exit(1)

    model.cuda()

    return model

normalize = pth_transforms.Compose([
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])