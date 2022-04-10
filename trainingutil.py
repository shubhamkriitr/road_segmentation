import os
import logging
from argparse import ArgumentParser
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datautil import (DataLoaderUtilFactory)
from model_factory import ModelFactory
from commonutil import get_timestamp_str, BaseFactory
from loggingutil import logger
from cost_functions import CostFunctionFactory


class BaseTrainer(object):

    def __init__(self, model: nn.Module, dataloader, cost_function,
                 optimizer: Optimizer,
                batch_callbacks=[], epoch_callbacks=[], config={}) -> None:
        self.model = model
        self.cost_function = cost_function
        self.dataloader = dataloader
        self.batch_callbacks = batch_callbacks
        self.epoch_callbacks = epoch_callbacks

        # read from config: TODO
        self.num_epochs = 100
        self.config = config
        self.num_epochs = self.config["num_epochs"]
        self.optimizer = optimizer


    def train(self):
        global_batch_number = 0
        current_epoch_batch_number = 0
        for current_epoch in range(1, self.num_epochs + 1):
            current_epoch_batch_number = 0
            for batch_data in self.dataloader:
                global_batch_number += 1
                current_epoch_batch_number += 1

                # perform one training step
                self.training_step(batch_data, global_batch_number,
                                    current_epoch, current_epoch_batch_number)
            self.invoke_epoch_callbacks(self.model, batch_data, global_batch_number,
                                current_epoch, current_epoch_batch_number)
            
    def training_step(self, batch_data,  global_batch_number, current_epoch,
                    current_epoch_batch_number):
        
        # make one training step
        
        raise NotImplementedError()

    def invoke_epoch_callbacks(self, model, batch_data, global_batch_number,
                                current_epoch, current_epoch_batch_number):
        self.invoke_callbacks(self.epoch_callbacks, 
                    [self.model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number], {})

    def invoke_callbacks(self, callbacks, args: list, kwargs: dict):
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                logger.exception(exc)

class NetworkTrainer(BaseTrainer):

    def training_step(self, batch_data, global_batch_number,
                        current_epoch, current_epoch_batch_number):
        # make sure training mode is on 
        self.model.train()

        # reset optimizer
        self.optimizer.zero_grad()

        # unpack batch data
        x, y_true = batch_data

        # compute model prediction
        y_pred = self.model(x)

        # compute loss
        loss = self.cost_function(input=y_pred, target=y_true)

        # backward pass
        loss.backward()

        # take optimizer step
        self.optimizer.step()

        self.invoke_callbacks(self.batch_callbacks, 
                    [self.model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number], {"loss": loss})


class BaseExperimentPipeline(object):
    """Class to link experiment stages like
    training, logging, evaluation, summarization etc.
    """

    def __init__(self, config) -> None:
        self.config = None
        self.initialize_config(config)
    
    def initialize_config(self, config):
        config = self.load_config(config)

        # TODO: add/ override some params here
        self.config = config


    def prepare_experiment(self):
        raise NotImplementedError()

    def run_experiment(self):
        raise NotImplementedError()

    def load_config(self, config):
        if isinstance(config, dict):
            return config
        if isinstance(config, str):
            config_data = {}
            with open(config, "r", encoding="utf-8") as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
            return config_data

# dictionary to refer to class by name
# (to be used in config)
TRAINER_NAME_TO_CLASS_MAP = {
    "NetworkTrainer": NetworkTrainer
}

# Factory class to get trainer class by name
class TrainerFactory(BaseFactory):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.resource_map = TRAINER_NAME_TO_CLASS_MAP
     
    def get(self, trainer_name, config=None,
            args_to_pass=[], kwargs_to_pass={}):
        return super().get(trainer_name, config,
                            args_to_pass=[], kwargs_to_pass={})

# TODO: may move optimizer part to another file
OPTIMIZER_NAME_TO_CLASS_OR_INITIALIZER_MAP = {
    "Adam": Adam,
    "AdamW": AdamW
}
class OptimizerFactory(BaseFactory):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.resource_map = OPTIMIZER_NAME_TO_CLASS_OR_INITIALIZER_MAP
    
    def get(self, optimizer_name, config=None,
                args_to_pass=[], kwargs_to_pass={}):
        return super().get(optimizer_name, config,
            args_to_pass, kwargs_to_pass)

class ExperimentPipeline(BaseExperimentPipeline):
    def __init__(self, config) -> None:
        super().__init__(config)

    def prepare_experiment(self):
        self.prepare_model()
        self.prepare_optimizer() # call this after model has been initialized
        self.prepare_scheduler()
        self.prepare_cost_function()
        self.prepare_summary_writer()
        self.prepare_dataloaders()
        self.prepare_batch_callbacks()
        self.prepare_epoch_callbacks()

        self.trainer = self.prepare_trainer()


    def prepare_dataloaders(self):
        dataloader_util_class_name = self.config["dataloader_util_class_name"]
        train_batch_size = self.config["batch_size"]

        train_loader, val_loader, test_loader \
        = DataLoaderUtilFactory()\
            .get(dataloader_util_class_name, config=None)\
            .get_data_loaders(root_dir=self.config["dataloader_root_dir"],
                              batch_size=train_batch_size,
                              shuffle=self.config["shuffle"]
                              normalize=self.config["normalize"])
            
        

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

    def prepare_trainer(self):
        trainer_class = TrainerFactory().get(
            self.config["trainer_class_name"])
        
        trainer = trainer_class(model=self.model,
                    dataloader=self.train_loader,
                    cost_function=self.cost_function,
                    optimizer=self.optimizer,
                    batch_callbacks=self.batch_callbacks,
                    epoch_callbacks=self.epoch_callbacks,
                    config={
                        "num_epochs": self.config["num_epochs"]
                        }
                    )

        self.trainer = trainer
        return trainer
    

    def prepare_model(self):
        # TODO: use model config too (or make it work by creating new class)
        model = ModelFactory().get(self.config["model_class_name"])()
        self.model = model

        if self.config["load_from_checkpoint"]:
            checkpoint_path = self.config["checkpoint_path"]
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
            logger.info(str(self.model))
            logger.info(f"Model Loaded")
        
        return self.model
    
    def prepare_optimizer(self):
        trainable_params, trainable_param_names, frozen_params, \
                frozen_param_names = self.filter_trainer_parameters()
        logger.info(f"Frozen Parameters: {frozen_param_names}")
        logger.info(f"Trainable Parameters: {trainable_param_names} ")
        lr = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]
        # TODO: Discuss and Add subfields (2nd level nesting) in the experminet 
        # config (yaml files) to pass args and kwargs if needed 
        self.optimizer = OptimizerFactory().get(
            self.config["optimizer_class_name"],
            config=None,
            args_to_pass=[],
            kwargs_to_pass={
                "lr": lr,
                "weight_decay": weight_decay,
                "params": trainable_params
            }
        )
    
    def prepare_scheduler(self):
        if "scheduler" not in self.config:
            return
        scheduler_name = self.config["scheduler"]
        if scheduler_name is None:
            return
        if scheduler_name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        else:
            raise NotImplementedError()
        
    
    def filter_trainer_parameters(self):
        trainable_params = []
        trainable_param_names = []
        frozen_params = []
        frozen_param_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                trainable_param_names.append(name)
            else:
                frozen_params.append(param)
                frozen_param_names.append(name)
        
        return trainable_params, trainable_param_names, frozen_params, \
                frozen_param_names

    def prepare_summary_writer(self):
        experiment_tag = self.config["experiment_metadata"]["tag"]
        timestamp = get_timestamp_str()
        self.current_experiment_directory = os.path.join(
            self.config["logdir"],timestamp+"_"+experiment_tag)

        os.makedirs(self.current_experiment_directory, exist_ok=True)
        self.current_experiment_log_directory = os.path.join(
            self.current_experiment_directory, "logs"
        )
        os.makedirs(self.current_experiment_log_directory, exist_ok=True)
        
        self.summary_writer = SummaryWriter(
            log_dir=self.current_experiment_log_directory)

    def prepare_cost_function(self):
        class_weights = self.prepare_class_weights_for_cost_function()
        kwargs_to_pass = {}
        if class_weights is not None:
            kwargs_to_pass["weight"] = class_weights
        
        self.cost_function = CostFunctionFactory().get(
            self.config["cost_function_class_name"],
            config=None,
            args_to_pass=[],
            kwargs_to_pass=kwargs_to_pass
        )
    
    def prepare_class_weights_for_cost_function(self):
        # TODO: Add if needed
        return None
        

    def prepare_batch_callbacks(self):
        self.batch_callbacks = [self.batch_callback]

    def prepare_epoch_callbacks(self):
        self.epoch_callbacks = [self.epoch_callback]

    def run_experiment(self):
        self.trainer.train()
    
    def batch_callback(self, model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number, **kwargs):
        
        if global_batch_number % self.config["batch_log_frequency"] == 0:
            logger.info(
            f"[({global_batch_number}){current_epoch}-{current_epoch_batch_number}]"
            f" Loss: {kwargs['loss']}")
        if global_batch_number % self.config["tensorboard_log_frequency"] == 0:
            self.summary_writer.add_scalar("train/loss", kwargs['loss'],
                                           global_batch_number)
    
    def epoch_callback(self, model: nn.Module, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number, **kwargs):
        if current_epoch == 1: # the epoch just finished
            # save the config
            self.save_config()
    
        model.eval()

    def save_config(self):
        try:
            file_path = os.path.join(self.current_experiment_directory,
                                    "config.yaml")
            with open(file_path, 'w') as f:
                yaml.dump(self.config, f)
        except Exception as exc:
            logger.exception(exc)       



class ExperimentPipelineForSegmentation(ExperimentPipeline):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.best_metric = None

    def epoch_callback(self, model: nn.Module, batch_data, global_batch_number,
     current_epoch, current_epoch_batch_number, **kwargs):
        if current_epoch == 1: # the epoch just finished
            # save the config
            self.save_config()
            with torch.no_grad():
                self.summary_writer.add_graph(self.model, batch_data[0])
    
        model.eval()
        # 

        with torch.no_grad():
            val_f1, _, _, _ = self.compute_and_log_evaluation_metrics(
                model, current_epoch, "val")
            test_f1, _, _, y_test_pred_prob = self.compute_and_log_evaluation_metrics(
                model, current_epoch, "test")
        
        metric_to_use_for_model_selection = val_f1 # TODO: can be pulled in config
        metric_name = "Validation F1-Score"
        if self.best_metric is None or \
             (self.best_metric < metric_to_use_for_model_selection):
            logger.info(f"Saving model: {metric_name} changed from {self.best_metric}"
                  f" to {metric_to_use_for_model_selection}")
            self.best_metric = metric_to_use_for_model_selection
            file_path = os.path.join(self.current_experiment_directory,
            "best_model.ckpt")
            torch.save(model.state_dict(), file_path)

        if hasattr(self, "scheduler"):
            self.scheduler.step(metric_to_use_for_model_selection)
            next_lr = [group['lr'] for group in self.optimizer.param_groups][0]
            self.summary_writer.add_scalar("lr", next_lr,
             current_epoch)
        return self.best_metric

    def compute_and_log_evaluation_metrics(self, model, current_epoch,
        eval_type):
        model.eval()
        eval_loss = 0.
        n_epochs = self.config["num_epochs"]
        with torch.no_grad():
            predictions = []
            targets = []

            for i, (inp, target) in enumerate(self.val_loader):
                # move input to cuda if required
                if self.config["device"] == "cuda": 
                    # TODO: take device info from `resolve_device`
                    inp = inp.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                # forward pass
                pred = model.forward(inp)
                loss = self.cost_function(pred, target)
                eval_loss += loss.item()
                predictions.append(pred)
                targets.append(target)

            targets = torch.cat(targets, axis=0)
            predictions = torch.cat(predictions, axis=0)
            f1_value = f1_score(predictions.to('cpu'), targets.int().to('cpu')[:, 0])

        self.summary_writer.add_scalar(
            f"{eval_type}/loss", eval_loss / len(self.val_loader.dataset),
            current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/F1", f1_value,
                                       current_epoch)
        logger.info(f"Evaluation loss after epoch {current_epoch}/{n_epochs}:"
                    f" {eval_loss / len(self.val_loader.dataset)}")
        logger.info(
            f"F1-Score after epoch {current_epoch}/{n_epochs}: {f1_value}")
        self.summary_writer.flush()
        return f1_value, loss
    
    
        
        

        
PIPELINE_NAME_TO_CLASS_MAP = {
    "ExperimentPipeline": ExperimentPipeline,
    "ExperimentPipelineForSegmentation": ExperimentPipelineForSegmentation
}


if __name__ == "__main__":
    DEFAULT_CONFIG_LOCATION = "experiment_configs/exp_00_sample.yaml"
    argparser = ArgumentParser()
    argparser.add_argument("--config", type=str,
                            default=DEFAULT_CONFIG_LOCATION)
    args = argparser.parse_args()
    
    config_data = None
    with open(args.config, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    

    pipeline_class = PIPELINE_NAME_TO_CLASS_MAP[ config_data["pipeline_class"]]
    pipeline = pipeline_class(config=config_data)
    pipeline.prepare_experiment()
    pipeline.run_experiment()


