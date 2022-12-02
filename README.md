# Examples
```
PYTHONPATH=./ python phraseology/model/train.py --mode tune --problem classification --model mlp --ds static_idioms --max-epochs 400;
PYTHONPATH=./ python phraseology/model/train.py --mode tune --problem classification --model mlp --ds formal_idioms --max-epochs 400;
PYTHONPATH=./ python phraseology/model/train.py --mode tune --problem classification --model mlp --ds idioms --max-epochs 400;
PYTHONPATH=./ python phraseology/model/train.py --mode tune --problem classification --model mlp --ds phrasal_verbs --max-epochs 400;
PYTHONPATH=./ python phraseology/model/train.py --mode tune --problem classification --model mlp --ds all --max-epochs 400;
```