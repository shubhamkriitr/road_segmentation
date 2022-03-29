from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import unet
import torch
import argparse
from datautil import *
from cost_functions import *
import baseline_unet
from torchmetrics import F1Score

def get_model_from_name(model_name="unet", model_config={}):
    if model_name == "unet":
        return unet.UNet(**model_config)
    elif model_name == "baseline_unet":
        return baseline_unet.BaselineUNet(**model_config)
    else:
        raise Exception("Model name not recognized. You should use one of the following: unet, baseline_unet")

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
          logging_freq=100):
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
    f1_score = F1Score()

    for epoch in range(n_epochs):
        epoch_loss = 0
        model.train()

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

                writer.add_scalar("Loss/eval", eval_loss / len(test_dataloader.dataset), epoch)
                print(f"Evaluation loss after epoch {epoch + 1}/{n_epochs}: {eval_loss / len(test_dataloader.dataset)}")
                print(f"F1-Score after epoch {epoch + 1}/{n_epochs}: {f1_score(predictions.to('cpu'), targets.int().to('cpu'))}")

        writer.flush()

    return model


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the paths
    train_data_path = None
    model_save_path = None
    logs_save_path = None
    assert train_data_path is not None, "Please specify the path to the training data"

    train_dataloader, test_dataloader = get_train_test_dataloaders(train_data_path, train_split=0.8)

    # Define the model
    model = get_model_from_name(model_name="unet")

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Choose a loss
    loss = torch.nn.CrossEntropyLoss()

    model = train(model,
                  loss_fn=loss,
                  optimizer=optimizer,
                  n_epochs=20,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  model_save_path=model_save_path,
                  logs_save_path=logs_save_path,
                  save_freq=None,
                  logging_freq=10,
                  device=device)

