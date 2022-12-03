# Installation
```
conda create -n bert python=3.9
conda activate bert
```

```
MACOS:
conda install pytorch torchvision torchaudio -c pytorch
LINUX:
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

```
conda install scikit-learn pandas

pip install tqdm apto animus simpletransformers path
```

# Examples
```
PYTHONPATH=./ python phraseology/model/train.py --mode tune --model mlp --ds all --problem classification --max-epochs 100
PYTHONPATH=./ python phraseology/model/train.py --mode experiment --model mlp --ds all --problem classification --max-epochs 100
```

```
PYTHONPATH=./ python phraseology/model/train_bert.py --mode experiment --problem classification --prefix test --max-epochs 100
```
```
PYTHONPATH=./ python phraseology/model/train_bert_sec.py --mode experiment --problem classification --prefix test --max-epochs 100
```
```
PYTHONPATH=./ python phraseology/model/train_regbert.py --mode experiment --problem regression --prefix test --max-epochs 100
```
```
PYTHONPATH=./ python phraseology/model/train_regbert_sec.py --mode experiment --problem regression --prefix test --max-epochs 100
```