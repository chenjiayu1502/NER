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
import model_on_crf as M
import evaluate as E
# from modelTrainer import BaseLSTMCRFTrainer
from trainer_on_crf2 import BaseLSTMCRFTrainer

from pl import get_perf



word_dim=[128]
lstm_dim=128
lstm_layers=1
dropout_prob=0.5
bidirectional=True
batch_size=128
shuffle=False
epoch=20
'''
input_path='data.txt'
label_path='label.txt'
input_test_path='data_test.txt'
label_test_path='label_test.txt'
'''

# input_path='./data0520/x_train_0622.txt'
input_path='./data0520/x_train_qu.txt'
label_path='./data0520/y_train_0622.txt'
# input_test_path='./data0520/x_test_0622.txt'
input_test_path='./data0520/x_test_qu.txt'
label_test_path='./data0520/y_test_0622.txt'

# input_path='./data0520/x_train.txt'
# label_path='./data0520/y_train.txt'
# input_test_path='./data0520/x_test.txt'
# label_test_path='./data0520/y_test.txt'
# input_path='./data0520/new_x_train.txt'
# label_path='./data0520/new_y_train.txt'
# input_test_path='./data0520/new_x_test.txt'
# label_test_path='./data0520/new_y_test.txt'

# input_test_path='./data0520/new_x_train.txt'
# label_test_path='./data0520/new_y_train.txt'


def create_dataloader(dataset):
    return D.MultiSentWordDataLoader(
        dataset=dataset,
        input_vocabs=input_vocabs,
        label_vocabs=label_vocab,
        batch_size=batch_size,
        shuffle=shuffle,
        tensor_lens=True,
        num_workers=1,
        pin_memory=True
    )


if __name__ == "__main__":
	input_vocabs = []
	input=input_path
	vocab = utils.Vocabulary()
	words = utils.FileReader(input).words()
	vocab.add("<pad>")
	vocab.add("<unk>")
	utils.populate_vocab(words, vocab)
	input_vocabs.append(vocab)
	print(input_vocabs)#[<utils.Vocabulary object at 0x7fa839f5a0b8>]
	print(len(vocab))

	label_vocab = utils.Vocabulary()
	words = utils.FileReader(label_path).words()
	label_vocab.add("START")
	label_vocab.add("END")
	utils.populate_vocab(words, label_vocab)

	crf = M.CRF(len(label_vocab))
	model = M.LSTMCRF(
	        crf=crf,
	        vocab_sizes=[len(v) for v in input_vocabs],
	        word_dims=word_dim,
	        hidden_dim=lstm_dim,
	        layers=lstm_layers,
	        dropout_prob=dropout_prob,
	        bidirectional=bidirectional
	    )
	# model.reset_parameters()
	# model=torch.load('./pkl/0613_multi_long_model_lstmlstm3.pkl')
	model2=M.LSTMLSTM(
		model=model,
		)
	# model2.reset_parameters()
	# model2=torch.load('./pkl/0605_multi_long_model_lstmlstm9.pkl')
	params = sum(np.prod(p.size()) for p in model.parameters())
	print("Number of parameters: {}".format(params))
	print(model2)

	print("Loading word embeddings...")

	dataset = D.MultiSentWordDataset(input_path, label_path)
	test_dataset = D.MultiSentWordDataset(input_test_path, label_test_path)
	train_dataset = dataset
	train_dataloader = create_dataloader(train_dataset)
	test_dataloader = create_dataloader(test_dataset)
	# model2=torch.load('./pkl/0623_multi_long_model_lstmlstm9.pkl')

	trainer=BaseLSTMCRFTrainer(
		model=model2,
		epochs=epoch,
		input_vocabs=input_vocabs,
		label_vocab=label_vocab
		)
	
	trainer.train(
		train_dataloader, 
		data_size=len(train_dataset)
		)
	
	
	#trainer=torch.load('./pkl/1.pkl')
	
	trainer.test(
		test_dataloader, 
		data_size=len(test_dataset)
		)
	accuracy, precision, recall, f1score=get_perf('./result/qu_long_pred_lstmlstm.txt')
	# torch.save(model,'./pkl/multi_long_model_lstmcrf.pkl')
	


	
	




