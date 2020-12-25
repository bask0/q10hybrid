
<div align="center">

# Deep Learning for Earth Sciences (DL4ES)

<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)

ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/bask0/dl4es/workflows/CI%20testing/badge.svg?branch=master&event=push)

<!--
Conference   
-->

</div>

## Description

A collection of spatio-temporal models, optionally using self-supervision.

## How to run

First, install dependencies

```bash
# clone project   
git clone https://github.com/bask0/dl4es

# install project   
cd dl4es
pip install -e .   
pip install -r requirements.txt
```

Next, navigate to any file and run it.

```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python tests/test_regression.py    
```

## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Imports

Happy to receive feedback and collaborate with you!

### Citation

```tex
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
