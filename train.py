from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import unet
import torch
import argparse
from datautil import *
from cost_functions import *
import baseline_unet
import efficient_unet
from torchmetrics import F1Score
import imageio
from model_resnet50 import(PrunedResnet50, get_pruned_resnet_50)

parser = argparse.ArgumentParser()

parser.add_argument("--data", "-d", default=None, type=str, help="path to training data")
parser.add_argument("--savedir", "-s", default=None, type=str, help="path for saving checkpoints")
parser.add_argument("--logdir", "-l", default=None, type=str, help="path for writing tensorboard logs")
parser.add_argument("--model", "-m", default="unet", type=str, help="what type of model should be trained (e.g. unet)")
parser.add_argument(
    "--save_images",
    "-i",
    action="store_true",
    help="flag specifying that dev data predictions should be saved to .png images"
)
parser.add_argument("--threshold", default=0.5, type=float, help="probability threshold for being marked as a road")

def get_model_from_name(model_name="unet", model_config={}):
    # TODO: create a choice list/dict instead
    model_choices = ["unet", "baseline_unet", "efficient_unet",
                     "pruned_resnet50"]
    if model_name == model_choices[0]:
        return unet.UNet(**model_config)
    elif model_name == model_choices[1]:
        return baseline_unet.BaselineUNet(**model_config)
    elif model_name == model_choices[2]:
        return efficient_unet.EfficientUNet()
    elif model_name == model_choices[3]:
        return get_pruned_resnet_50()
    else:
        raise NameError(f"Model name not recognized. You should"
                        f"use one of the following: {model_choices}")

def train(model: torch.nn.Module,
          loss_fn: callable,
          optimizer: callable,
          n_epochs: int,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          model_save_path: str,
          logs_save_path: str,
          save_freq: int = None,
          device: str = "cpu",
          logging_freq=100,
          initial_epochs=0,
          threshold=0.5):
    """
    Function used to call a model for the CIL project
    :param model: Model to be trained
    :param loss: function used to compute the loss. Takes as input (prediction, target)
    :param optimizer: optimizer for the model from torch.optim
    :param n_epochs: number of epochs to train for
    :param batch_size: training batch size
    :param train_dataloader: Dataloader containing training data
    :param test_dataloader: Dataloader containing testing data
    :param model_save_path: Path where model checkpoints will be stored
    :param logs_save_path: Path where training logs will be stored
    :param save_freq: Frequency used to store the model
    :param device: Device used for training. Options: "cuda" or "cpu"
    """

    # Define a run id for the execution
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"Starting training a {model} architecture with id {run_id} on {device}")

    # Create storing folders if they do not exist
    for d in [logs_save_path, model_save_path]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Define a summary writer for the logs
    writer = SummaryWriter(os.path.join(logs_save_path, run_id))

    model = model.to(device)

    # Training process
    f1_score = F1Score(threshold=threshold)

    for epoch in range(n_epochs):
        epoch_loss = 0
        model.train()
        if epoch >= initial_epochs and type(model) == efficient_unet.EfficientUNet:
            model.update_enet = True

        for i, (inp, target) in enumerate(train_dataloader):

            # move input to cuda if required
            if device == "cuda":
                inp = inp.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            if epoch == 0 and i == 0:
                writer.add_graph(model, inp)

            # forward pass
            pred = model.forward(inp)
            loss = loss_fn(pred, target)

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # optimizer step
            optimizer.step()

            # log
            epoch_loss += loss.item()

            if i % logging_freq == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} | Batch {i}/{len(train_dataloader)} => Running epoch loss per sample: {epoch_loss / (logging_freq * train_dataloader.batch_size)}")

        # Log the loss
        writer.add_scalar("Loss/train", epoch_loss / len(train_dataloader.dataset), epoch)

        # Store model
        if save_freq is not None and ((epoch + 1) % save_freq == 0 or epoch + 1 == n_epochs):
            checkpoint_name = f"{run_id}_epoch{epoch + 1}"
            checkpoint_path = os.path.join(model_save_path, f"{checkpoint_name}.pkl")
            print(f"Saving model to {checkpoint_path}")
            torch.save(model, checkpoint_path)

        if test_dataloader:
            model.eval()
            eval_loss = 0.

            with torch.no_grad():
                predictions = []
                targets = []

                for i, (inp, target) in enumerate(test_dataloader):
                    # move input to cuda if required
                    if device == "cuda":
                        inp = inp.cuda(non_blocking=True)
                        target = target.cuda(non_blocking=True)

                    # forward pass
                    pred = model.forward(inp)
                    loss = loss_fn(pred, target)
                    eval_loss += loss.item()
                    predictions.append(pred)
                    targets.append(target)

                targets = torch.cat(targets, axis=0)
                predictions = torch.cat(predictions, axis=0)
                f1_value = f1_score(predictions.to('cpu'), targets.int().to('cpu')[:, 0])

                writer.add_scalar("Loss/eval", eval_loss / len(test_dataloader.dataset), epoch)
                writer.add_scalar("F1", f1_value, epoch)
                print(f"Evaluation loss after epoch {epoch + 1}/{n_epochs}: {eval_loss / len(test_dataloader.dataset)}")
                print(f"F1-Score after epoch {epoch + 1}/{n_epochs}: {f1_value}")

        writer.flush()

    return model

def write_images(model, dataloader, path, threshold):
    if not os.path.exists(path): os.makedirs(path)
    for b, (x, y) in enumerate(dataloader):
        pred = None
        with torch.no_grad():
            pred = model(x.cuda()).cpu()
        thresholded = torch.where(pred >= threshold, torch.ones_like(pred), torch.zeros_like(pred)).numpy()
        pred = pred.numpy()
        y = y.cpu().numpy()[:, :1]
        assert pred.shape == y.shape
        for i in range(pred.shape[0]):
            imageio.imwrite(f"{path}/{b}-{i}.png", (pred[i, 0]*255).astype(np.uint8))
            imageio.imwrite(f"{path}/{b}-{i}_thresholded.png", (thresholded[i, 0]*255).astype(np.uint8))
            imageio.imwrite(f"{path}/{b}-{i}_label.png", (y[i, 0]*255).astype(np.uint8))

# TODO: move weighed BCE to cost_functions.py or remove?
base_bce = torch.nn.BCELoss(reduction='none')
def weighted_BCELoss(y_pred, y_true):
    y_true = y_true[:, :1]
    int_loss = base_bce(y_pred, y_true)
    pred_round = y_pred.detach().round()
    weights = torch.where((pred_round==0) & (y_true==1), 5, 1)
    return torch.mean(weights*int_loss)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the paths
    assert args.data is not None, "Please specify the path to the training data"

    train_dataloader, test_dataloader = get_train_test_dataloaders(args.data, train_split=0.8)

    # Define the model
    model = get_model_from_name(model_name=args.model)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Choose a loss
    loss = weighted_BCELoss

    model = train(model,
                  loss_fn=loss,
                  optimizer=optimizer,
                  n_epochs=100,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  model_save_path=args.savedir,
                  logs_save_path=args.logdir,
                  save_freq=None,
                  logging_freq=10,
                  device=device,
                  initial_epochs=20,
                  threshold=args.threshold)

    if args.save_images:
        write_images(model, train_dataloader, "./images/train", args.threshold)
        write_images(model, test_dataloader, "./images/dev", args.threshold)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
