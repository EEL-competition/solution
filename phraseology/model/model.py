# pylint: disable=C0115,C0103,C0116,R1725,R0913,E0401,W0212
"""Models for experiments"""
import os
import json
import pandas as pd
import torch
from torch import nn, optim
from scipy.stats import randint, uniform, loguniform
from phraseology.model.settings import LOGS_ROOT
from phraseology.model.train import Experiment


def get_config(exp: Experiment):
    config = {}
    model_config = {}

    if exp.mode == "experiment":
        # find and load the best tuned model
        runs_files = []

        searched_dir = exp.project_name.split("-")
        searched_dir = "-".join(searched_dir[1:])
        serached_dir = f"tune-{searched_dir}"
        if exp.project_prefix != exp.utcnow:
            serached_dir = f"{exp.project_prefix}-{serached_dir}"
        print(f"Searching trained model in {LOGS_ROOT}*{serached_dir}")
        for logdir in os.listdir(LOGS_ROOT):
            if logdir.endswith(serached_dir):
                runs_files.append(os.path.join(LOGS_ROOT, logdir))

        # if multiple runs files found, choose the latest
        runs_file = sorted(runs_files)[-1]
        print(f"Using best model from {runs_file}")

        # get general config
        with open(f"{runs_file}/config.json", "r") as fp:
            config = json.load(fp)

        # get model config
        df = pd.read_csv(f"{runs_file}/runs.csv", delimiter=",", index_col=False)
        # pick hyperparams of a model with the highest test_score
        best_config_path = df.loc[df["test_score"].idxmax()].to_dict()
        best_config_path = best_config_path["config_path"]
        with open(best_config_path, "r") as fp:
            model_config = json.load(fp)

        # leave only necessary keys
        config = {
            "max_epochs": config["max_epochs"],
            "batch_size": config["batch_size"],
            "num_epochs": config["num_epochs"],
        }

        print("Loaded coonfigs:")
        print(config)
        print(model_config)

    elif exp.mode == "tune":
        # add the link to the wandb run
        model_config["link"] = exp.wandb_logger.get_url()
        # set batch size
        config["batch_size"] = randint.rvs(
            4, min(32, int(exp.data_shape[0] / exp.n_splits) - 1)
        )
        # set learning rate
        model_config["lr"] = loguniform.rvs(1e-5, 1e-3)

        # pick random hyperparameters
        if exp.model == "mlp":
            model_config["hidden_size"] = randint.rvs(32, 256)
            model_config["num_layers"] = randint.rvs(0, 4)
            model_config["dropout"] = uniform.rvs(0.1, 0.9)
            model_config["input_size"] = exp.data_shape[1]

            if exp.problem == "classification":
                model_config["output_size"] = exp.n_classes
            elif exp.problem == "regression":
                model_config["output_size"] = 1
        elif exp.model == "wide_mlp":
            model_config["hidden_size"] = randint.rvs(256, 1024)
            model_config["num_layers"] = randint.rvs(0, 4)
            model_config["dropout"] = uniform.rvs(0.1, 0.9)
            model_config["input_size"] = exp.data_shape[1]

            if exp.problem == "classification":
                model_config["output_size"] = exp.n_classes
            elif exp.problem == "regression":
                model_config["output_size"] = 1

        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    config["num_epochs"] = exp.max_epochs
    config["runpath"] = f"{exp.logdir}k_{exp.k}/{exp.trial:04d}"
    config["run_config_path"] = f"{exp.logdir}k_{exp.k}/{exp.trial:04d}/config.json"

    return config, model_config


def get_model(model, model_config):
    if model in ["mlp", "wide_mlp"]:
        return MLP(model_config)

    raise NotImplementedError()


def get_criterion(model, problem):
    if model in ["mlp", "wide_mlp"]:
        if problem == "classification":
            return nn.CrossEntropyLoss()
        if problem == "regression":
            return nn.CrossEntropyLoss()

    raise NotImplementedError()


def get_optimizer(exp: Experiment):
    optimizer = optim.Adam(
        exp._model.parameters(),
        lr=float(exp.model_config["lr"]),
    )

    return optimizer


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        input_size = int(config["input_size"])
        output_size = int(config["output_size"])
        dropout = float(config["dropout"])
        hidden_size = int(config["hidden_size"])
        num_layers = int(config["num_layers"])

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # bs, fs = x.shape
        # bs:  batch size
        # fs:  feature_size

        output = self.fc(x)
        return output
