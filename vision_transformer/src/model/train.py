import json
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm

import torch
from torch import nn

import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.tensorboard import SummaryWriter
from torchvision import transforms as pth_transforms

from torchmetrics import F1Score

from .utils import *

def train(model, 
          classifier, 
          train_loader, 
          validation_loader, 
          log_dir=None,
          optimizer=None, 
          criterion=nn.CrossEntropyLoss(),
          epochs=5, 
          val_freq=1, 
          batch_size=16,
          to_restore = {"epoch": 0, "best_acc": 0.}, 
          n=4, 
          avgpool_patchtokens=False):

    """ Trains a classifier ontop of a base model. The input can be perturbed by selecting an adversarial attack.
        
        :param model: base model (frozen)
        :param classifier: classifier to train
        :param train_loader: loader of the train set
        :param validation_loader: dataloader of the validation dataset
        :param log_dir: path to the log directory.
        :param tensor_dir: if set saves the output of the model in the dir
        :param optimizer: optimizer for the training process. Default: None -> uses the SGD as defined by DINO.
        :param adversarial_attack: adversarial attack for adversarial training. Default: None -> the classifier is trained without adversarial perturbation.
        :param epochs: number of epochs to train the classifier on. Default: 5
        :param val_freq: frequency (in epochs) in which the classifier is validated.
        :param batch_size: batch_size for training and validation. Default: 16
        :param lr: the learning rate of the optimizer if the DINO optimizer is used. Default: 0.001
        :param to_restore:
        :param n: from DINO. Default: 4
        :param avgpool_patchtokens: from DINO. Default: False
        
    """
    if model is None or classifier is None:
        raise Exception("You must provide a ViT and a Classifier")

    # Move to CUDA
    model.cuda()
    classifier.cuda()

    if optimizer is None:
        optimizer = torch.optim.SGD([{'params': classifier.parameters(),
            "lr": 1e-3, # linear scaling rule
            "momentum":0.9,
            "weight_decay":0.},
            {'params': model.parameters(), "lr": 5e-6}
        ], lr=1e-3)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    
    # Optionally resume from a checkpoint
    log_dir.mkdir(parents=True, exist_ok=True)
    restart_from_checkpoint(
        Path(log_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    
    # train loop
    loggers = {'train':[], 'validation':[]}
    for epoch in range(start_epoch, epochs):
        if 'set_epoch' in dir(train_loader.sampler):
            train_loader.sampler.set_epoch(epoch)

        # train epoch
        train_stats, metric_logger = train_epoch(model=model, 
                                                 classifier=classifier, 
                                                 optimizer=optimizer, 
                                                 criterion=criterion,
                                                 train_loader=train_loader,
                                                 epoch=epoch, 
                                                 n=n, 
                                                 avgpool_patchtokens=avgpool_patchtokens)
        loggers['train'].append(metric_logger)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        
        # validate
        if epoch % val_freq == 0 or epoch == epochs - 1:
            test_stats, metric_logger = validate_network(model=model, 
                                                         classifier=classifier, 
                                                         validation_loader=validation_loader, 
                                                         criterion=criterion,
                                                         n=n, 
                                                         avgpool_patchtokens=avgpool_patchtokens)

            loggers['validation'].append(metric_logger)
            print(f"Accuracy at epoch {epoch} of the network on the {len(validation_loader)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        # log
        if is_main_process():
            with (Path(log_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, Path(log_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
    return loggers
    
    


def train_epoch(model, 
                classifier, 
                train_loader, 
                optimizer, 
                criterion=nn.CrossEntropyLoss(),
                epoch=0, 
                n=4, 
                avgpool_patchtokens=False):
    """ Trains a classifier ontop of a base model. The input can be perturbed by selecting an adversarial attack.
        
        :param model: base model (frozen)
        :param classifier: classifier to train
        :param optimizer: optimizer for the training process.
        :param train_loader: dataloader of the train dataset
        :param tensor_dir: if set saves the output of the model in the dir

        :param adversarial_attack: adversarial attack for adversarial training. Default: None -> the classifier is trained without adversarial perturbation.
        :param epochs: The current epch
        :param n: from DINO. Default: 4
        :param avgpool_patchtokens: from DINO. Default: False
        
    """
    classifier.train()
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    if len(train_loader)<20:
        log_interval = 1
    elif len(train_loader)<100:
        log_interval = 5
    else:
        log_interval = 20   
    
    for (inp, target, names) in metric_logger.log_every(train_loader, log_interval, header):
        
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        if model is not None:
            # forward
            with torch.no_grad():
                model_output = model_forward(model, inp, n, avgpool_patchtokens)
            
            output = classifier(model_output)
            
        else:
            output = classifier(inp)
            
        # compute loss
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger


def validate_network(model, 
                     classifier, 
                     validation_loader, 
                     criterion=nn.CrossEntropyLoss(),
                     n=4, 
                     avgpool_patchtokens=False,
                     log_interval=None):
    """ Validates a classifier ontop of an optional model with an optional 
        adversarial attack. 
        
        :param model: base model (frozen)
        :param classifier: classifier to train
        :param validation_loader: dataloader of the validation dataset
        :param criterion: The loss criterion. Default: CrossEntropyLoss
        :param tensor_dir: if set saves the output of the model in the dir.
        :param adversarial_attack: adversarial attack for adversarial training.
                                   Default: None -> the classifier is trained 
                                   without adversarial perturbation.
        :param n: from DINO. Default: 4
        :param avgpool_patchtokens: from DINO. Default: False
        :param path_predictions: If given, saves a csv file at path_predictions
                                 containing the filenames, the true label, the 
                                 prediction and if there is an adversarial 
                                 attack the adversarial prediction.
        :param show_image: shows the last couple images in the last batch.
    """
    if model is not None:
        model.eval()
    classifier.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    f1score = F1Score()
    
    if log_interval is None:
        if len(validation_loader)<20:
            log_interval = 1
        elif len(validation_loader)<100:
            log_interval = 5
        else:
            log_interval = 20
    for inp, target, batch_names in metric_logger.log_every(validation_loader, log_interval, header):

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True) 

        # forward
        with torch.no_grad():
            model_output = model_forward(model, inp, n, avgpool_patchtokens)
            output = classifier(model_output)
            loss = criterion(output, target)

        # compute f1-score
        f1s = f1score(output.cpu(), target.cpu())

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['f1s'].update(f1s, n=batch_size)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger


def model_forward(model, inp, n=4, avgpool_patchtokens=False):
    """ Performs a forward pass on a dino model.
        
        :param model: dino model (frozen)
        :param inp: the input for the model
        :param n: from DINO. Default: 4
        :param avgpool_patchtokens: from DINO. Default: False
    """
    
    # Normalize
    transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    inp = transform(inp)  

    if 'get_intermediate_layers' in dir(model):
        intermediate_output = model.get_intermediate_layers(inp, n)
        model_output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if avgpool_patchtokens:
            model_output = torch.cat((model_output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            model_output = model_output.reshape(model_output.shape[0], -1)
    else:
        model_output = model(inp)
    return model_output
