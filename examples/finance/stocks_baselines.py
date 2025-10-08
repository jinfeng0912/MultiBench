from torch import nn
import torch.nn.functional as F
import torch
# import pmdarima  # Not actually used in this script
import numpy as np
import argparse
import sys
import os
sys.path.append(os.getcwd())

from datasets.stocks.get_data import get_dataloader # noqa
from unimodals.common_models import LSTM # noqa
from fusions.common_fusions import Stack # noqa



parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(
    stocks, stocks, [args.target_stock], modality_first=True)


def baselines():
    def best_constant(y_prev, y):
        return float(nn.MSELoss()(torch.ones_like(y) * torch.mean(y), y))

    def copy_last(y_prev, y):
        return nn.MSELoss()(torch.cat([y_prev[-1:], y[:-1]]), y).item()

    # ARIMA 依赖 pmdarima，出于兼容性默认跳过
    def arima(y_prev, y):
        return float('nan')

    # Resolve datasets
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    # test_loader is a dict of lists of DataLoaders; pick clean (noise_level=0)
    test_ds = test_loader['timeseries'][0].dataset

    print('Best constant val MSE loss: ' +
          str(best_constant(train_ds.Y, val_ds.Y)))
    print('Best constant test MSE loss: ' +
          str(best_constant(val_ds.Y, test_ds.Y)))
    print('Copy-last val MSE loss: ' +
          str(copy_last(train_ds.Y, val_ds.Y)))
    print('Copy-last test MSE loss: ' +
          str(copy_last(val_ds.Y, test_ds.Y)))
    print('ARIMA val MSE loss: skipped (pmdarima disabled)')
    print('ARIMA test MSE loss: skipped (pmdarima disabled)')


baselines()
