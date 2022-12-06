import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import string
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

from sklearn.model_selection import KFold

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from transformers import BertModel, BertTokenizer

from transformers import BertTokenizer, TFBertModel

from transformers import pipeline

import gc

from tensorflow.keras.callbacks import EarlyStopping

from phraseology.model.settings import LOGS_ROOT

sns.set(style="darkgrid", font_scale=1)


def preprocess(text):

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@[0-9a-zA-Z]*\W+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\#", " ", text)
    text = re.sub(r"\'", " ", text)

    list_text = text.split()
    text = " ".join(list_text[:512])
    return text


def encode(input_text, max_len):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer.batch_encode_plus(
        input_text, padding="max_length", max_length=max_len, truncation=True
    )
    return inputs


def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=-1, keepdims=True)


def create_model(max_len):
    bert_encoder = TFBertModel.from_pretrained("bert-base-uncased")
    input_word_ids = tf.keras.Input(
        shape=(max_len,), dtype=tf.int32, name="input_word_ids"
    )

    embedding = bert_encoder(input_word_ids)[0]
    x = tf.keras.layers.GlobalAveragePooling1D()(embedding)
    x = tf.keras.layers.LayerNormalization()(x)
    # Output layer without activation function because regression task
    output = tf.keras.layers.Dense(
        6,
    )(x)

    model = tf.keras.models.Model(inputs=input_word_ids, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=MCRMSE, metrics=MCRMSE)

    return model


def train():
    train_df = pd.read_csv("../raw_data/train.csv")

    targets = [
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions",
    ]

    all_data = train_df
    all_data.drop(["text_id"], axis=1, inplace=True)
    print("all_data size is : {}".format(all_data.shape))

    PUNCT_TO_REMOVE = string.punctuation
    STOPWORDS = set(stopwords.words("english"))

    lemmatizer = WordNetLemmatizer()

    all_data["full_text"] = all_data["full_text"].apply(lambda text: preprocess(text))

    kf = KFold(n_splits=10)

    tr_idx = []
    test_idx = []
    for train_index, test_index in kf.split(all_data):
        tr_idx.append(train_index)
        test_idx.append(test_index)
        # train_data = all_data.loc[train_index].copy()
        # test_data = all_data.loc[test_index].copy()

    for trial in range(10):
        train_data = all_data.loc[tr_idx[trial]].copy()
        test_data = all_data.loc[test_idx[trial]].copy()

        BATCH_SIZE = 6

        MAX_LEN = max(len(x.split()) for x in all_data["full_text"])

        AUTO = tf.data.experimental.AUTOTUNE

        unmasker = pipeline("fill-mask", model="bert-base-uncased")

        train_input = encode(train_data["full_text"].values.tolist(), MAX_LEN)[
            "input_ids"
        ]

        train_data_ds = (
            tf.data.Dataset.from_tensor_slices(
                (train_input, train_data.drop("full_text", axis=1))
            )
            .repeat()
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )

        testing_input = encode(test_data.full_text.values.tolist())["input_ids"]

        test_dataset = tf.data.Dataset.from_tensor_slices(testing_input).batch(
            BATCH_SIZE
        )

        model = create_model(MAX_LEN)

        gc.collect()

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="MCRMSE", patience=2, restore_best_weights=True
        )

        history = model.fit(
            train_data_ds,
            steps_per_epoch=train_data.shape[0] // BATCH_SIZE,
            batch_size=BATCH_SIZE,
            epochs=3,
            verbose=1,
            shuffle=True,
            callbacks=[callback],
        )

        results = model.predict(test_dataset)

        test_data.to_csv(f"{LOGS_ROOT}/bert/true_{trial}.csv", index=False)
        results.to_csv(f"{LOGS_ROOT}/bert/pred_{trial}.csv", index=False)
