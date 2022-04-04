from datautil import *
from cost_functions import *
from vision_transformer.src.model.classification_head import BaselineUNet
from vision_transformer.src.model.dino_model import get_dino
from vision_transformer.src.model.train import train

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the paths
    train_data_path = None
    model_save_path = None
    logs_save_path = None
    assert train_data_path is not None, "Please specify the path to the training data"

    train_dataloader, test_dataloader = get_train_test_dataloaders(train_data_path, train_split=0.8)

    # Define the model
    classifier = BaselineUNet()
    model = get_dino(model_name='vit_small', patch_size=8)

    # Define the optimizer
    optimizer = torch.optim.SGD([{'params': classifier.parameters(),
            "lr": 1e-3, # linear scaling rule
            "momentum":0.9,
            "weight_decay":0.},
            {'params': model.parameters(), "lr": 1e-4}
        ])

    # Choose a loss
    loss = BinaryGeneralizeDiceLossV2()

    train(model = model,
              classifier=classifier,
              train_loader=train_dataloader,
              validation_loader=test_dataloader,
              log_dir=None,
              tensor_dir=None,
              optimizer=optimizer,
              criterion=loss,
              epochs=50,
              val_freq=10,
              batch_size=16)
