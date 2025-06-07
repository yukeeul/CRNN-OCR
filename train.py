import os
import subprocess

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger

from src.model import PLCRNN
from src.utils import Dataset, Tokenizer, load_dataset


@hydra.main(version_base=None, config_path="configs", config_name="main.yaml")
def train(cfg):
    img_size = cfg["model"]["img_size"]
    dataset = load_dataset(cfg["data"]["load_from_dvc"])

    train_dataset = Dataset(dataset["train"], img_size=img_size)
    test_dataset = Dataset(dataset["test"], img_size=img_size)

    tokenizer = Tokenizer(**cfg["tokenizer"])
    model = PLCRNN(
        charset_size=cfg["model"]["charset_size"], hidden_size=cfg["model"]["hidden_size"], tokenizer=tokenizer
    )

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    mlflow_logger = MLFlowLogger(
        tracking_uri="http://localhost:8080",
        tags={
            "git_commit": git_commit,
            "author": "Iurii Ulianov",
        },
    )
    mlflow_logger.log_hyperparams(cfg)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"])

    trainer = pl.Trainer(
        limit_train_batches=cfg["training"]["num_train_batches"],
        max_epochs=cfg["training"]["epochs"],
        limit_val_batches=cfg["training"]["num_val_batches"],
        val_check_interval=1.0,
        logger=mlflow_logger,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    experiment_id = mlflow_logger.run_id
    torch.save(model.state_dict(), f"models/{experiment_id}.pt")

    example_inputs = (torch.randn(1, 1, *img_size),)
    torch.onnx.export(model, example_inputs, f"models/{experiment_id}.onnx")

    print("")
    print(f"Trained model saved to models/{experiment_id}.onnx")
    print("Pushing to dvc ...")
    print("")

    os.system(f"dvc add models/{experiment_id}.onnx")
    os.system("dvc push -r models")


if __name__ == "__main__":
    train()
