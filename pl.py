#encoding=utf-8
import os
import sys
import stat
import subprocess
import time

import torch
from torch.autograd import Variable
import torch.autograd as autograd
#import torchtext.data as data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('..')
#import config
import utils



# metrics function using conlleval.pl
def conlleval(p, g, w, filename):
    """
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    """
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += str(w + ' ' + wl + ' ' + wp + '\n')
        out += 'EOS O O\n\n'
    # print(out)

    f = open(filename, 'w')
    f.writelines(out[:-1])  # remove the ending \n on last line
    f.close()

    return get_perf(filename)


def get_perf(filename):
    """ run conlleval.pl perl script to obtain
    precision/recall and F1 score """
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.pl'
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                             _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(b''.join([line.encode() for line in open(filename).readlines()]))
    for line in stdout.split(b'\n'):
        if b'accuracy' in line:
            out = line.split()
            break
    accuracy = float(out[1][:-2])
    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])
    print(accuracy, precision, recall, f1score)
    return accuracy, precision, recall, f1score


if __name__ =='__main__':
    accuracy, precision, recall, f1score=get_perf('./result/0617_long_pred_lstmcrf.txt')


