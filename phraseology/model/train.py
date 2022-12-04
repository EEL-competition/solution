# pylint: disable=W0201,W0223,C0103,C0115,C0116,R0902,E1101,R0914,E0401, C0415
"""Script for tuning different models on different datasets"""
import os
import csv
import sys
import argparse
import json
import re
import shutil

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm.auto import tqdm

from apto.utils.report import get_classification_report

from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import TorchCheckpointerCallback

from apto.utils.misc import boolean_flag

import wandb

from phraseology.model.settings import LOGS_ROOT, UTCNOW
from phraseology.model.dataset import load_dataset


class Experiment(IExperiment):
    def __init__(
        self,
        mode: str,
        path: str,
        problem_type: str,
        model: str,
        dataset: str,
        balanced: bool,
        prefix: str,
        n_splits: int,
        n_trials: int,
        max_epochs: int,
    ) -> None:
        super().__init__()
        self.config = {}

        self.utcnow = self.config["default_prefix"] = UTCNOW
        # starting fold/trial; used in resumed experiments
        self.start_k = 0
        self.start_trial = 0

        if mode == "resume":
            (
                mode,
                problem_type,
                model,
                dataset,
                n_splits,
                n_trials,
                max_epochs,
                prefix,
            ) = self.acquire_cont_params(path, n_trials)

        # tune or experiment mode
        self.mode = self.config["mode"] = mode
        # classification or regression
        self.problem = self.config["problem"] = problem_type
        # model name
        self.model = self.config["model"] = model
        # main dataset name (used for training and testing)
        self._dataset = self.config["dataset"] = dataset
        self.balanced = self.config["balanced"] = balanced

        # num of splits for StratifiedKFold
        self.n_splits = self.config["n_splits"] = n_splits
        # num of trials for each fold
        self.n_trials = self.config["n_trials"] = n_trials

        self.max_epochs = self.config["max_epochs"] = max_epochs

        # set project name prefix
        if prefix is None:
            self.project_prefix = f"{self.utcnow}"
        else:
            if len(prefix) == 0:
                self.project_prefix = f"{self.utcnow}"
            else:
                # '-'s are reserved for name parsing
                self.project_prefix = prefix.replace("-", "_")
        self.config["prefix"] = self.project_prefix

        if self.balanced:
            self.project_name = f"{self.mode}-{self.model}-balanced_{self._dataset}"
        else:
            self.project_name = f"{self.mode}-{self.model}-{self._dataset}"

        self.config["project_name"] = self.project_name
        self.logdir = f"{LOGS_ROOT}/{self.project_prefix}-{self.project_name}/"
        self.config["logdir"] = self.logdir

        # create experiment directory
        os.makedirs(self.logdir, exist_ok=True)
        # save initial config - used in 'resume' mode if it is ever needed
        logfile = f"{self.logdir}/config.json"
        with open(logfile, "w") as fp:
            json.dump(self.config, fp)

    def acquire_cont_params(self, path, n_trials):
        """
        Used for extracting experiments set-up from the
        given path for continuing an interrupted experiment
        """
        # load experiment config
        with open(f"{path}/config.json", "r") as fp:
            config = json.load(fp)

        # find when the experiment got interrupted
        with open(path + "/runs.csv", "r") as fp:
            lines = len(fp.readlines()) - 1
            self.start_k = lines // n_trials
            self.start_trial = lines - self.start_k * n_trials

        # delete failed run
        faildir = path + f"/k_{self.start_k}/{self.start_trial:04d}"
        print("Deleting interrupted run logs in " + faildir)
        try:
            shutil.rmtree(faildir)
        except FileNotFoundError:
            print("Could not delete interrupted run logs - FileNotFoundError")

        self.utcnow = self.config["default_prefix"] = config["default_prefix"]

        return (
            config["mode"],
            config["problem"],
            config["model"],
            config["dataset"],
            config["n_splits"],
            config["n_trials"],
            config["max_epochs"],
            config["prefix"],
        )

    def initialize_data(self):
        # load dataset
        # your dataset should have shape [n_features; phraseology space]

        features, labels = load_dataset(self._dataset, self.problem)
        self.data_shape = features.shape  # [n_features; phraseology space]
        print("data shape: ", self.data_shape)
        if self.problem == "classification":
            self.n_classes = len([*set(labels)])
            print("number of classes: ", self.n_classes)

        # train-val/test split
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        skf.get_n_splits(features, labels)

        train_index, test_index = list(skf.split(features, labels))[self.k]

        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.data_shape[0] // self.n_splits,
            random_state=42 + self.trial,
            stratify=y_train,
        )

        if self.balanced:
            print(f"X_train shape before balancing: {X_train.shape}")
            print(f"y_train shape before balancing: {y_train.shape}")

            filter_labels = y_train.copy()
            if self.problem == "regression":
                filter_labels = (filter_labels * 2 - 2).astype(int)

            filter_array = []

            label_count = np.zeros(np.unique(filter_labels).shape[0])
            for i, label in enumerate(filter_labels):
                if label_count[label] < 100:
                    label_count[label] += 1
                    filter_array.append(i)

            X_train, y_train = X_train[filter_array], y_train[filter_array]
            print(f"X_train shape after balancing: {X_train.shape}")
            print(f"y_train shape after balancing: {y_train.shape}")

        self._train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.int64),
        )
        self._valid_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.int64),
        )
        self._test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.int64),
        )

    def initialize_model(self):
        # deferred import - there is a cyclic import
        from phraseology.model.model import (
            get_config,
            get_model,
            get_criterion,
            get_optimizer,
        )

        config, self.model_config = get_config(self)
        self.config.update(config)
        self._model = get_model(self.model, self.model_config)
        self.criterion = get_criterion(self.model, self.problem)
        self.optimizer = get_optimizer(self)

        # create run directory
        os.makedirs(config["runpath"], exist_ok=True)

        # update saved config
        with open(f"{self.logdir}/config.json", "w") as fp:
            json.dump(self.config, fp)
        # save model params in the run's directory
        with open(self.config["run_config_path"], "w") as fp:
            json.dump(self.model_config, fp)

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)

        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project=f"{self.project_prefix}-{self.project_name}",
            name=f"{self.utcnow}-k_{self.k}-trial_{self.trial}",
            save_code=True,
        )

        # init data
        self.initialize_data()

        # init model: config, model, loss function, optimizer
        self.initialize_model()

        # setup data loaders
        self.datasets = {
            "train": DataLoader(
                self._train_ds,
                batch_size=int(self.config["batch_size"]),
                num_workers=0,
                shuffle=True,
            ),
            "valid": DataLoader(
                self._valid_ds,
                batch_size=int(self.config["batch_size"]),
                num_workers=0,
                shuffle=False,
            ),
        }

        # setup callbacks
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=True,
                patience=30,
                dataset_key="valid",
                metric_key="loss",
                min_delta=0.001,
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="_model",
                logdir=self.config["runpath"],
                dataset_key="valid",
                metric_key="loss",
                minimize=True,
            ),
        }

        # set epochs
        self.num_epochs = self.max_epochs

        self.wandb_logger.config.update(self.config)
        self.wandb_logger.config.update(self.model_config)

    def run_dataset(self) -> None:
        all_scores, all_targets = [], []
        total_loss = 0.0
        self._model.train(self.is_train_dataset)

        if self.problem == "classification":
            with torch.set_grad_enabled(self.is_train_dataset):
                for self.dataset_batch_step, (data, target) in enumerate(
                    tqdm(self.dataset)
                ):
                    self.optimizer.zero_grad()
                    logits = self._model(data)
                    loss = self.criterion(logits, target)
                    score = torch.softmax(logits, dim=-1)

                    all_scores.append(score.cpu().detach().numpy())
                    all_targets.append(target.cpu().detach().numpy())
                    total_loss += loss.sum().item()
                    if self.is_train_dataset:
                        loss.backward()
                        self.optimizer.step()

            total_loss /= self.dataset_batch_step + 1

            y_test = np.hstack(all_targets)
            y_score = np.vstack(all_scores)
            y_pred = np.argmax(y_score, axis=-1).astype(np.int32)

            report = get_classification_report(
                y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
            )

            self.dataset_metrics = {
                "accuracy": report["precision"].loc["accuracy"],
                "score": report["auc"].loc["weighted"],
                "loss": total_loss,
            }

            if self.dataset_key.startswith("test"):
                logpath = f"{self.logdir}k_{self.k}/{self.trial:04d}/confusion_matrix_data.npz"
                np.savez(logpath, y_true=y_test, y_pred=y_pred, y_score=y_score)

        if self.problem == "regression":
            pass

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(self)
        if self.problem == "classification":
            self.wandb_logger.log(
                {
                    "train_accuracy": self.epoch_metrics["train"]["accuracy"],
                    "train_score": self.epoch_metrics["train"]["score"],
                    "train_loss": self.epoch_metrics["train"]["loss"],
                    "valid_accuracy": self.epoch_metrics["valid"]["accuracy"],
                    "valid_score": self.epoch_metrics["valid"]["score"],
                    "valid_loss": self.epoch_metrics["valid"]["loss"],
                },
            )

        if self.problem == "regression":
            pass

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)

        # prepare to run test: load test dataset
        self.dataset_key = "test"
        self.dataset = DataLoader(
            self._test_ds,
            batch_size=int(self.config["batch_size"]),
            num_workers=0,
            shuffle=False,
        )

        # prepare to run test: load best model weights
        logpath = f"{self.logdir}k_{self.k}/{self.trial:04d}/_model.best.pth"
        checkpoint = torch.load(logpath, map_location=lambda storage, loc: storage)
        self._model.load_state_dict(checkpoint)

        print("Run test dataset")
        self.run_dataset()

        # save results
        if self.problem == "classification":
            print("Test results:")
            print("Accuracy ", self.dataset_metrics["accuracy"])
            print("AUC ", self.dataset_metrics["score"])
            print("Loss ", self.dataset_metrics["loss"])

            results = {
                "test_accuracy": self.dataset_metrics["accuracy"],
                "test_score": self.dataset_metrics["score"],
                "test_loss": self.dataset_metrics["loss"],
                "config_path": self.config["run_config_path"],
            }

            self.wandb_logger.log(results)

            df = pd.DataFrame(results, index=[0])
            with open(f"{self.logdir}/runs.csv", "a") as f:
                df.to_csv(f, header=f.tell() == 0, index=False)

        if self.problem == "regression":
            pass

        self.wandb_logger.finish()

    def tune(self):
        folds_of_interest = []

        if self.start_k != self.n_splits:
            for trial in range(self.start_trial, self.n_trials):
                folds_of_interest += [(self.start_k, trial)]
            for k in range(self.start_k + 1, self.n_splits):
                for trial in range(self.n_trials):
                    folds_of_interest += [(k, trial)]
        else:
            raise IndexError()

        for k, trial in folds_of_interest:
            self.k = k  # k'th test fold
            self.trial = trial  # trial'th trial on the k'th fold
            self.run()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    for_resume = "resume" in sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tune",
            "experiment",
            "resume",
        ],
        required=True,
        help="'tune' for model hyperparams tuning; 'experiment' \
            for experiments with tuned model; 'resume' for resuming \
                interrupted experiment",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=for_resume,
        help="path to the interrupted experiment (e.g., \
            /Users/user/mlp_project/assets/logs/prefix-mode-model-ds)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        choices=[
            "regression",
            "classification",
        ],
        required=not for_resume,
        help="'regression' for regression problem,\
            'classification' for classification approach",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "mlp",
            "wide_mlp",
        ],
        required=not for_resume,
        help="Name of the model to run",
    )
    parser.add_argument(
        "--ds",
        type=str,
        choices=[
            "all",
            "phrasal_verbs",
            "idioms",
            "formal_idioms",
            "static_idioms",
        ],
        required=not for_resume,
        help="Name of the dataset to use for training",
    )

    boolean_flag(parser, "balanced", default=False)

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for the project name (body of the project name \
            is '$mode-$problem-$model-$dataset'): default: UTC time",
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials to run on each test fold",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="Number of splits for StratifiedKFold (affects the number of test folds)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Max number of epochs (min 30)",
    )
    args = parser.parse_args()

    Experiment(
        mode=args.mode,
        path=args.path,
        problem_type=args.problem,
        model=args.model,
        dataset=args.ds,
        balanced=args.balanced,
        prefix=args.prefix,
        n_splits=args.num_splits,
        n_trials=args.num_trials,
        max_epochs=args.max_epochs,
    ).tune()
