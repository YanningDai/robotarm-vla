import json
from collections import defaultdict
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from scipy.special import lambertw
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm



delta = 0.07
z = -np.exp(-(delta + 1.0))
c_low = float(-lambertw(z, k=0).real)
c_high = float(-lambertw(z, k=-1).real)
            
print(f"c_low: {c_low}, c_high: {c_high}")  