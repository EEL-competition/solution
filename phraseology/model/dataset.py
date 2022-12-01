# pylint: disable=C0115,C0103,C0116,R1725,R0913,E0401
"""Script loading dataset"""
import numpy as np

from phraseology.model.settings import DATA_ROOT  # , PREPROCESSED_ROOT


def load_dataset(dataset: str, problem_type: str):
    """
    Return the dataset defined by type
    """
    if dataset == "static_idioms":
        data = load_static_idioms()
    elif dataset == "formal_idioms":
        data = load_formal_idioms()
    elif dataset == "idioms":
        data1 = load_static_idioms()
        data2 = load_formal_idioms()
        data = np.concatenate((data1, data2), axis=1)
    elif dataset == "phrasal_verbs":
        data = load_phrasal_verbs()
    elif dataset == "all":
        data1 = load_static_idioms()
        data2 = load_formal_idioms()
        data3 = load_phrasal_verbs()
        data = np.concatenate((data1, data2, data3), axis=1)
    else:
        raise NotImplementedError()

    if problem_type == "classification":
        labels = load_labels()
    elif problem_type == "regression":
        labels = load_regression_labels()
    else:
        raise NotImplementedError()

    return data, labels


def load_static_idioms():
    path = DATA_ROOT.joinpath("static_idioms_vector.npy")
    data = np.load(path)

    return data


def load_formal_idioms():
    path = DATA_ROOT.joinpath("formal_idioms_vector.npy")
    data = np.load(path)

    return data


def load_phrasal_verbs():
    path = DATA_ROOT.joinpath("phrasal_verbs_vector.npy")
    data = np.load(path)

    return data


def load_labels():
    path = DATA_ROOT.joinpath("labels.npy")
    labels = np.load(path)

    labels = labels * 2
    labels -= 2
    labels = labels.astype("int")

    return labels


def load_regression_labels():
    path = DATA_ROOT.joinpath("labels.npy")
    labels = np.load(path)

    return labels
