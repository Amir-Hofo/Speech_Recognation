import os
import math
import warnings

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torchaudio
from torchaudio import transforms as T
from torchtext.vocab import build_vocab_from_iterator

from torchvision import models

from torchmetrics.text import WordErrorRate as WER
from torchmetrics.aggregation import MeanMetric