#encoding=utf-8
import os
import pickle
import shutil
import logging
import argparse
import tempfile
import subprocess
import collections

import numpy as np
#import yaap
import tqdm
import torch
import torch.nn as nn
import torch.optim as O
import torch.autograd as A
import sys

import utils
import data as D
import model4 as M
#import model2 as M
import evaluate as E


def beam_search_decoder(lens_s,p_array,k,):
	pass


if __name__ == "__main__":
	# define a sequence of 10 words over a vocab of 5 words
	data = [[0.1, 0.2, 0.3, 0.4, 0.5],
	        [0.5, 0.4, 0.3, 0.2, 0.1],
	        [0.1, 0.2, 0.3, 0.4, 0.5],
	        [0.5, 0.4, 0.3, 0.2, 0.1],
	        [0.1, 0.2, 0.3, 0.4, 0.5],
	        [0.5, 0.4, 0.3, 0.2, 0.1],
	        [0.1, 0.2, 0.3, 0.4, 0.5],
	        [0.5, 0.4, 0.3, 0.2, 0.1],
	        [0.1, 0.2, 0.3, 0.4, 0.5],
	        [0.5, 0.4, 0.3, 0.2, 0.1]]