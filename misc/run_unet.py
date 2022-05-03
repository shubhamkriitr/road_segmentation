from unet import *
from train import train

if __name__ == "__main__":
    print(f"will train on {device}")
    train_dataloader, test_dataloader = get_train_test_dataloaders("./data/training", train_split=0.8)

    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model = train(model,
                  loss_fn=torch.nn.CrossEntropyLoss(),
                  optimizer=optimizer,
                  n_epochs=20,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  model_save_path="./data/checkpoints/unet",
                  logs_save_path="./data/checkpoints/unet",
                  save_freq=None,
                  logging_freq=1,
                  device=device)

    test_image, test_target = test_dataloader.dataset[3]
    pred = model(test_image)[0]
    pred = pred.round()

    imageio.imwrite("./output_mask.png", pred)

