import os
from argparse import ArgumentParser
import torch
import yaml
from torchmetrics import F1Score, Accuracy, Precision, Recall
from torch import nn
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.datautil import (DataLoaderUtilFactory)
from models.model_factory import ModelFactory
from utils.commonutil import get_timestamp_str, BaseFactory, read_config
from utils import commonutil
from utils.loggingutil import logger
from training.cost_functions import CostFunctionFactory
from submit.mask_to_submission import python_execution as mask_to_submission
import copy
import numpy as np
import PIL

"""This file contains most of the code implementing the pipeline behavior. The most commonly used is ExperimentPipelineForSegmentation and EnsemblePipeline. This code is also responsible for interpreting the config file and writing to logs."""

def merge_dicts(initial, override):
    ret = {}
    for key in initial.keys(): ret[key] = initial[key]
    for key in override.keys(): ret[key] = override[key]
    return ret

class BaseTrainer(object):

    def __init__(self, model: nn.Module, dataloader, cost_function,
                 optimizer: Optimizer,
                 batch_callbacks=[], epoch_callbacks=[], config={}) -> None:
        self.model = model
        try:
            self.model_event_handler = self.model.handle_event
        except AttributeError:
            self.model_event_handler = None
            pass # TODO: warn
        self.cost_function = cost_function
        self.dataloader = dataloader
        self.batch_callbacks = batch_callbacks
        self.epoch_callbacks = epoch_callbacks

        # read from config: TODO
        self.num_epochs = 100
        self.config = config
        self.num_epochs = self.config["num_epochs"]
        self.optimizer = optimizer

    def process_events(self, current_epoch):
        if self.model_event_handler is None: return
        try:
            for event in self.config["events"][current_epoch]:
                self.model_event_handler(event)
        except KeyError: # no events for the current epoch
            pass

    def train(self):
        global_batch_number = 0
        current_epoch_batch_number = 0
        self.process_events(current_epoch=0)
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
            self.process_events(current_epoch)

    def training_step(self, batch_data, global_batch_number, current_epoch,
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

        # unpack batch data and shift to resovled device (to cuda if available)
        x, y_true = (t.to(commonutil.resolve_device()) for t in batch_data)

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

    def __init__(self, config, overwrite_config={}) -> None:
        self.config = None
        self.initialize_config(config, overwrite_config)

    def initialize_config(self, config, overwrite_config={}):
        config = self.load_config(config)

        self.config = merge_dicts(config, overwrite_config)

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
    def __init__(self, config, overwrite_config={}) -> None:
        super().__init__(config, overwrite_config)

    def prepare_experiment(self):
        self.prepare_model()
        self.prepare_optimizer()  # call this after model has been initialized
        self.prepare_scheduler()
        self.prepare_cost_function()
        self.prepare_metrics()
        self.prepare_summary_writer()
        self.prepare_dataloaders()
        self.prepare_batch_callbacks()
        self.prepare_epoch_callbacks()

        self.trainer = self.prepare_trainer()

    def prepare_dataloaders(self):
        dataloader_util_class_name = self.config["dataloader_util_class_name"]
        train_batch_size = self.config["batch_size"]

        data_splits = self.config.get("ensemble_data_splits")
        dataloader_config = {"num_folds": data_splits} \
            if "ensemble_data_splits" in self.config else None
        train_loader, val_loader, test_loader \
            = DataLoaderUtilFactory() \
            .get(dataloader_util_class_name, config=dataloader_config) \
            .get_data_loaders(root_dir=self.config["data_root_dir"],
                              batch_size=train_batch_size,
                              shuffle=self.config["shuffle"],
                              normalize=self.config["normalize"])
        try:
            i = self.config["ensemble_dataloader_idx"]
            if data_splits is not None:
                assert len(train_loader) == data_splits, \
                    f"asked for {data_splits} dataloaders but got {len(train_loader)} of them"
            train_loader = train_loader[i]
        except KeyError:
            pass

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

    def prepare_trainer(self):
        trainer_class = TrainerFactory().get_uninitialized(
            self.config["trainer_class_name"])

        trainer = trainer_class(model=self.model,
                                dataloader=self.train_loader,
                                cost_function=self.cost_function,
                                optimizer=self.optimizer,
                                batch_callbacks=self.batch_callbacks,
                                epoch_callbacks=self.epoch_callbacks,
                                config=self.config
                                )

        self.trainer = trainer
        return trainer

    def prepare_model(self):
        # TODO: use model config too (or make it work by creating new class)
        model = ModelFactory().get(self.config["model_class_name"])
        self.model = model

        # use cuda if available (TODO: decide to use config/resolve device)
        self.model.to(commonutil.resolve_device())

        if self.config["load_from_checkpoint"]:
            checkpoint_path = self.config["checkpoint_path"]
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            self.model.load_state_dict(
                torch.load(checkpoint_path,
                           map_location=commonutil.resolve_device()))
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
        logger.info(f"Using optimizer: {self.optimizer}")

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
        logger.info(f"Using scheduler: {self.scheduler}")

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

        self.current_experiment_directory = self.config.get("experiment_directory") if self.config.get(
            "experiment_directory") else os.path.join(
            self.config["logdir"], timestamp + "_" + experiment_tag)

        os.makedirs(self.current_experiment_directory, exist_ok=True)
        self.current_experiment_log_directory = os.path.join(
            self.current_experiment_directory, "logs"
        )
        os.makedirs(self.current_experiment_log_directory, exist_ok=True)

        self.summary_writer = SummaryWriter(
            log_dir=self.current_experiment_log_directory)

        logger.info(self.current_experiment_directory)

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

    def prepare_metrics(self):
        self.metrics = {}
        self.metrics["F1"] = F1Score(num_classes=2, threshold=self.config["threshold"], average="weighted",
                                     multiclass=True)
        self.metrics["Accuracy"] = Accuracy(num_classes=2, threshold=self.config["threshold"], average="weighted",
                                            multiclass=True)
        self.metrics["Recall"] = Recall(num_classes=2, threshold=self.config["threshold"], average="weighted",
                                        multiclass=True)
        self.metrics["Precision"] = Precision(num_classes=2, threshold=self.config["threshold"], average="weighted",
                                              multiclass=True)

    def prepare_class_weights_for_cost_function(self):
        # TODO: Add if needed
        return None

    def prepare_batch_callbacks(self):
        self.batch_callbacks = [self.batch_callback]

    def prepare_epoch_callbacks(self):
        self.epoch_callbacks = [self.epoch_callback]

    def run_experiment(self):
        self.trainer.train()

        if self.config.get("create_submission"):
            self.create_submission()

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
        if current_epoch == 1:  # the epoch just finished
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

    def load_best_model_to(self, model_obj):
        best_file_name = [i for i in os.listdir(self.current_experiment_directory) if "best_model" in i][0]

        # Load best model
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.current_experiment_directory, best_file_name),
                map_location=commonutil.resolve_device()))

    def create_submission(self, model=None):
        logger.info("======== Creating submission file ========")

        if model is None:
            self.load_best_model_to(self.model)
        else:
            self.model = model

        # Save images
        output_dir = os.path.join(
            self.current_experiment_directory, "output_images"
        )
        logger.info(f"Saving test images at: {output_dir}")
        test_output_dir = (os.path.join(output_dir, "test"))

        os.makedirs(output_dir, exist_ok=False)
        commonutil.write_images(self.model, self.test_loader,
                                test_output_dir, self.config["threshold"],
                                save_submission_files=True, offset=144)

        # Create CSV file
        mask_to_submission(base_dir=os.path.join(output_dir, "submit/predictions"),
                           submission_filename=os.path.join(self.current_experiment_directory, "to_upload.csv"))
        logger.info(
            "========> Submision file exported to {}".format(os.path.join(test_output_dir, "submit/to_upload.csv")))


class ExperimentPipelineForSegmentation(ExperimentPipeline):
    def __init__(self, config, overwrite_config={}, evaluation=False) -> None:
        security_overwrite = {
            "experiment_directory": None # Ensure we do not overwrite existing experiment by mistake in training
        } if evaluation is False and not config.get("ensembling") else {}
        super().__init__(config, merge_dicts(overwrite_config, security_overwrite))
        self.best_metric = None

    def epoch_callback(self, model: nn.Module, batch_data, global_batch_number,
                       current_epoch, current_epoch_batch_number, **kwargs):
        if current_epoch == 1:  # the epoch just finished
            # save the config
            self.save_config()
            with torch.no_grad():
                self.summary_writer.add_graph(
                    self.model, batch_data[0].to(commonutil.resolve_device()))

        model.eval()
        #

        val_f1, val_prec, val_recall, val_acc, _ = self.compute_and_log_evaluation_metrics(
            model, current_epoch, "val")

        metric_to_use_for_model_selection = val_f1
        metric_name = "Validation F1-Score"

        if self.best_metric is None or \
                (self.best_metric < metric_to_use_for_model_selection):
            logger.info(f"Saving model: {metric_name} changed from "
                        f"{self.best_metric} to {metric_to_use_for_model_selection}")
            self.best_metric = metric_to_use_for_model_selection
            file_path = os.path.join(self.current_experiment_directory,
                                     f"best_model_{self.config['model_name_tag']}.ckpt")
            torch.save(model.state_dict(), file_path)
            self.save_probability_map(model=None, probability_map=None,
                current_epoch=current_epoch, type_="test", is_best=True)

        if (current_epoch % self.config["model_save_frequency"] == 0) \
                or (current_epoch == self.config["num_epochs"]):
            file_path = os.path.join(self.current_experiment_directory,
                                     f"model_{self.config['model_name_tag']}_" \
                                     + f"{str(current_epoch).zfill(4)}.ckpt")
            torch.save(model.state_dict(), file_path)
            self.save_probability_map(model=None, probability_map=None,
                current_epoch=current_epoch, type_="test", is_best=False)

        if hasattr(self, "scheduler"):
            self.scheduler.step(metric_to_use_for_model_selection)
            next_lr = [group['lr'] for group in self.optimizer.param_groups][0]
            self.summary_writer.add_scalar("lr", next_lr,
                                           current_epoch)

        # don't forget to dump log so far
        self.summary_writer.flush()

        # save images if asked:
        if (current_epoch == self.config["num_epochs"]) and \
                "save_images" in self.config and self.config["save_images"]:
            self.save_images()

        return self.best_metric

    def compute_and_log_evaluation_metrics(self, model, current_epoch,
                                           eval_type):
        pooling = torch.nn.AvgPool2d(kernel_size=16, stride=16)  # 16 is evaluation patch size
        model.eval()
        n_epochs = self.config["num_epochs"]
        _loader = None
        if eval_type == "val":
            _loader = self.val_loader
        elif eval_type == "test":
            _loader = self.test_loader
        else:
            raise AssertionError(f"Unsupported eval type: {eval_type}")
        predictions, targets, avg_eval_loss = self.evaluate_model(model, _loader)

        targets = torch.cat(targets, axis=0)
        predictions = torch.cat(predictions, axis=0)

        predictions_patches = pooling(predictions).to('cpu').flatten()
        targets_patches = pooling(targets).round().int().to('cpu')[:, 0].flatten()

        predictions = predictions.to('cpu').flatten()
        targets = targets.int().to('cpu')[:, 0].flatten()

        f1_value = self.metrics["F1"](
            predictions, targets)
        precision_value = self.metrics["Precision"](
            predictions, targets)
        acc_value = self.metrics["Accuracy"](
            predictions, targets)
        recall_value = self.metrics["Recall"](
            predictions, targets)

        f1_value_patches = self.metrics["F1"](
            predictions_patches, targets_patches)
        precision_value_patches = self.metrics["Precision"](
            predictions_patches, targets_patches)
        acc_value_patches = self.metrics["Accuracy"](
            predictions_patches, targets_patches)
        recall_value_patches = self.metrics["Recall"](
            predictions_patches, targets_patches)

        self.summary_writer.add_scalar(
            f"{eval_type}/loss", avg_eval_loss, current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/F1", f1_value,
                                       current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/Precision", precision_value,
                                       current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/Recall", recall_value,
                                       current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/Accuracy", acc_value,
                                       current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/F1_patches", f1_value_patches,
                                       current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/Precision_patches", precision_value_patches,
                                       current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/Recall_patches", recall_value_patches,
                                       current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/Accuracy_patches", acc_value_patches,
                                       current_epoch)
        logger.info(f"Evaluation loss after epoch {current_epoch}/{n_epochs}:"
                    f" {avg_eval_loss}")
        logger.info(
            f"""Evaluation metrics after epoch {current_epoch}/{n_epochs}:\nPixel-level => F1: {f1_value} Acc: {acc_value} | Precision: {precision_value} | Recall: {recall_value}\nPatch-level => F1: {f1_value_patches} Acc: {acc_value_patches} | Precision: {precision_value_patches} | Recall: {recall_value_patches}\n------------------------""")

        return f1_value, precision_value, recall_value, acc_value, avg_eval_loss

    def evaluate_model(self, model, _loader):
        eval_loss = 0. # It is cumulative eval loss
        with torch.no_grad():
            predictions = []
            targets = []

            for i, (inp, target) in enumerate(_loader):
                # move input to cuda if required
                # >>> if self.config["device"] == "cuda":
                # >>>     inp = inp.cuda(non_blocking=True)
                # >>>     target = target.cuda(non_blocking=True)
                inp, target = inp.to(commonutil.resolve_device()), \
                              target.to(commonutil.resolve_device())

                # forward pass
                try:
                    pred = model.forward(inp)
                except TypeError:
                    pred = model.forward(inp, i)
                loss = self.cost_function(pred, target)
                eval_loss += loss.item()
                predictions.append(pred)
                targets.append(target)
        avg_eval_loss = eval_loss / len(_loader.dataset)
        return predictions, targets, avg_eval_loss

    def save_images(self):
        output_dir = os.path.join(
            self.current_experiment_directory, "output_images"
        )
        logger.info(f"Saving images at: {output_dir}")
        train_output_dir, val_output_dir = (os.path.join(output_dir, p)
                                            for p in ["train", "val"])
        os.makedirs(output_dir, exist_ok=False)
        commonutil.write_images(self.model, self.train_loader,
                                train_output_dir, self.config["threshold"])
        commonutil.write_images(self.model, self.val_loader,
                                val_output_dir, self.config["threshold"])
        
    def save_probability_map(self, model, probability_map, current_epoch, type_, is_best):
        """ 
        `probability_map`: numpy array of probability maps of size (N, H, W),
            if it is passed as `None`, then using the current model state,
            predictions will be computed again.
        
        `current_epoch`: current epoch (to be used in the filename)
        
        `type_` is either `val` or `test`
        
        `is_best`: flag to indicate if it is the predictions by the best model
            so far
        
        """
        logger.debug(f"preparing to save probability map for type : `{type_}`")
        if model is None:
            logger.debug("Using current model state. (self.model)")
            model = self.model
        
        _loader = None
        if probability_map is None:
            if type_ == "val":
                _loader = self.val_loader
            elif type_ == "test":
                _loader = self.test_loader
            else:
                raise ValueError(f"Unsupported type : {type_}")

            logger.debug(f"Will compute predictions for inputs from "
                         f"dataloader: {_loader}")
            
            probability_map, _, _ = self.evaluate_model(model, _loader)
            probability_map = torch.concat(probability_map, dim=0)
            
            
        if isinstance(probability_map, torch.Tensor):
            probability_map = probability_map.cpu().numpy()
    
        tag = "pred-probability-maps"
        
        if is_best:
            output_file_name \
                = f"best-{type_}-{tag}"
        else:
            output_file_name \
                = f"{type_}-epoch-{str(current_epoch).zfill(4)}-{tag}"
        
        out_path = os.path.join(self.current_experiment_directory,
                                output_file_name)
        
        logger.info(f"Saving to file path: {out_path}")
        
        np.save(out_path, probability_map)

class EvaluationPipelineForSegmentation(ExperimentPipelineForSegmentation):
    def __init__(self, config, overwrite_config={}) -> None:
        assert config.get("experiment_directory") is not None, "experiment_directory to evaluate must be specified"
        assert "config.yaml" in os.listdir(
            config["experiment_directory"]), "experiment_directory must contain config.yaml"
        super().__init__(os.path.join(config['experiment_directory'], "config.yaml"),
                         merge_dicts(overwrite_config, {"experiment_directory": copy.deepcopy(config["experiment_directory"]),
                                             "pipeline_class": "EvaluationPipelineForSegmentation"}), evaluation=True)

    def run_experiment(self):
        # Print metrics on validation
        val_f1, val_prec, val_recall, val_acc, _ = self.compute_and_log_evaluation_metrics(
            self.model, 0, "val")
        self.create_submission()


PIPELINE_NAME_TO_CLASS_MAP = {
    "ExperimentPipeline": ExperimentPipeline,
    "ExperimentPipelineForSegmentation": ExperimentPipelineForSegmentation,
    "EvaluationPipelineForSegmentation": EvaluationPipelineForSegmentation
}


class Ensemble(torch.nn.Module):
    """Part of old ensembling implementation. Stores a number of models and averages their responses. It was replaced by a different method because the RAM size posed too severe limitations on the number of models ensembled."""
    def __init__(self, models=[]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
    def add_model(self, model):
        self.models.append(model)
    def forward(self, x):
        y = None
        with torch.no_grad():
            for model in self.models:
                if y is None:
                    y = model(x)
                else:
                    y += model(x)
        y /= len(self.models)
        return y


class ImgEnsemble(torch.nn.Module):
    """The newer ensemble implementation. Reads probability maps stored as images on the disk."""
    def __init__(self, models=[]):
        super().__init__()
        self.models = models
        self.dummy_params = nn.Conv1d(1, 1, 1) # some parts of the code assume the model has some weiths and crash otherwise
    def add_model(self, model):
        self.models.append(model)
    def forward(self, x, batch_num):
        i = 0
        batch = []
        first_path = self.models[0] + "/0-0.png"
        assert os.path.exists(first_path), f"nonexistent path: {first_path}"
        while os.path.exists(self.models[0] + f"/{batch_num}-{i}.png"):
            mask = None
            for m in self.models:
                model_mask = np.asarray(PIL.Image.open(m + f"/{batch_num}-{i}.png")) / 255
                if mask is None:
                    mask = model_mask
                else:
                    mask += model_mask
            mask /= len(self.models)
            batch.append(mask)
            i += 1
        if len(batch) != x.shape[0]:
            logger.warning(f"batch {batch_num}: expected batch size {len(batch)}, found input bs {x.shape[0]}")
        batch = batch[:x.shape[0]]
        return torch.tensor(np.stack(batch)).unsqueeze(1).cuda()


class EnsemblePipeline(ExperimentPipelineForSegmentation):
    """A pipeline for training a number of models and ensembling them. calls `run_experiment`, which is the main function called when the training starts."""
    def __init__(self, config):
        super().__init__(config)
        self.master_config = config
        self.master_config["ensembling"] = True

    def prepare_experiment(self):
        self.prepare_summary_writer()
        self.prepare_dataloaders()
        self.prepare_cost_function()
        self.prepare_metrics()
        self.prepare_dataloaders()
    
    def run_experiment(self):
        models = []
        img_dir_suffix = "/output_images/test"
        if "ensemble" in self.master_config:
            for i, conf_path in enumerate(self.master_config["ensemble"]):
                self.master_config["experiment_directory"] = self.current_experiment_directory + f"/{i}"
                conf_data = merge_dicts(self.master_config, read_config(conf_path))
                try:
                    override_dict = self.master_config["override"]
                    for key in override_dict.keys():
                        conf_data[key] = override_dict[key]
                except KeyError: # no override config
                    pass
                run_experiment(conf_data, ignore_ensemble=True)
                models.append(self.master_config["experiment_directory"] + img_dir_suffix)
        else:
            ensemble_data_splits = self.master_config["ensemble_data_splits"]
            for i in range(ensemble_data_splits):
                self.master_config["experiment_directory"] = self.current_experiment_directory + f"/{i}"
                logger.info(f"training network {i+1}/{ensemble_data_splits} for the split-ensemble")
                os.makedirs(self.current_experiment_directory, exist_ok=True)
                self.master_config["ensemble_dataloader_idx"] = i
                run_experiment(self.master_config, ignore_split_ensemble=True)
                models.append(self.master_config["experiment_directory"] + img_dir_suffix)
        ensemble = ImgEnsemble(models)
        self.compute_and_log_evaluation_metrics(ensemble, 0, "val")
        if self.master_config["create_submission"]: self.create_submission(model=ensemble)


# was originally part of run_experiment module but it is used for ensembling, which means trainingutil would
# have to import run_experiment, which would create a cyclic dependency
def run_experiment(config_data, return_model=False, ignore_ensemble=False, ignore_split_ensemble=False):
    config_data = config_data.copy()
    if ignore_ensemble: config_data.pop("ensemble", None)
    if "ensemble" in config_data or ("ensemble_data_splits" in config_data and not ignore_split_ensemble):
        pipeline_class = EnsemblePipeline
    else:
        pipeline_class = PIPELINE_NAME_TO_CLASS_MAP[config_data["pipeline_class"]]
    pipeline = pipeline_class(config=config_data)
    pipeline.prepare_experiment()
    pipeline.run_experiment()
    if return_model:
        model = pipeline.trainer.model
        pipeline.load_best_model_to(model)
        return pipeline.trainer.model

if __name__ == "__main__":
    DEFAULT_CONFIG_LOCATION = "experiment_configs/exp_03_resnet50_split.yaml"
    argparser = ArgumentParser()
    argparser.add_argument("--config", type=str,
                           default=DEFAULT_CONFIG_LOCATION)
    args = argparser.parse_args()

    config_data = None
    with open(args.config, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    pipeline_class = PIPELINE_NAME_TO_CLASS_MAP[config_data["pipeline_class"]]
    pipeline = pipeline_class(config=config_data)
    pipeline.prepare_experiment()
    pipeline.run_experiment()
