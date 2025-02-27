# Lightgbm
Based on 10-fold cross validation, the performance of model for correlation prediction of rice heading date traits

## System requirements
Python 3.10 / 3.11.

## Install Lightgbm environment

```bash
conda env create -f lightgbm.yml
```

## Run Lightgbm

```bash
python Lightgbm.py
```
Tips： Please note the modification of the file path.Due to the use of 10-fold cross-validation repeated 100 times, the training time may be considerable. To facilitate direct code execution by users, we also provide a version in Jupyter Notebook format.
