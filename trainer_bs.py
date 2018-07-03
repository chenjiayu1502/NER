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
import model_bs2 as M
import evaluate as E
from torch.optim.lr_scheduler import StepLR #as StepLR

import numpy as np



class BaseLSTMCRFTrainer(object):
    def __init__(self, model: M.LSTMCRF, epochs,input_vocabs,label_vocab, optimizer=O.SGD):
        self.model = model
        self.epochs = epochs
        self.input_vocabs = input_vocabs
        self.label_vocab = label_vocab
        self.optimizer_cls = optimizer
        self.optimizer = optimizer(self.model.parameters(),lr=0.8)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.gpu_main = None

        

    def wrap_var(self, x, **kwargs):
        x = A.Variable(x, **kwargs)
        return x

    @staticmethod
    def _ensure_tuple(x):
        if not isinstance(x, collections.Sequence):
            return (x,)

        return x

    def prepare_batch(self, xs, y, lens, **var_kwargs):
        lens, idx = torch.sort(lens, 0, True)
        xs, y = xs[:, idx], y[idx]
        xs, y = self.wrap_var(xs, **var_kwargs), self.wrap_var(y, **var_kwargs)
        lens = self.wrap_var(lens, **var_kwargs)

        return xs, y, lens


    def trans_probability(self,data):
        trans=np.zeros((len(self.label_vocab),len(self.label_vocab)),dtype=float)
        for i_idx, (batch, lens) in enumerate(data):
            #print(i_idx,(batch, lens))
            # print(lens)
            lenss=lens.numpy().tolist()
            pad=self.label_vocab.f2i["O"]
            # print('pad==',pad)
            xs, y = batch[:-1], batch[-1]
            # print(type(y),y.size())
            y=y.numpy()#.tolist()
            for i in range(len(y)):
                for j in range(lenss[i]-1):
                    fl=y[i][j]
                    bl=y[i][j+1]
                    trans[fl][bl]+=1.0
        for i in range(len(self.label_vocab)):
            lineall=np.sum(trans[i])
            if lineall>0:
                for j in range(len(self.label_vocab)):
                    trans[i][j]=trans[i][j]/lineall
                    
        return trans.tolist()

    def lexicalize_data(self, seq, vocab):
        return [vocab.i2f[w] if w in vocab else "<unk>" for w in seq]

    def lexicalize_label(self, seq, vocab):
        return [vocab.i2f[w] if w in vocab else 'O' for w in seq]

    @staticmethod
    def tighten(seqs, lens):
        return [s[:l] for s, l in zip(seqs, lens)]

    def compute_label_acc(self,pred,targ):
        assert len(pred)==len(targ)
        #print(len(pred))
        acc=0.0
        all=0.001
        for i in range(len(pred)):
            #print(len(pred[i]))
            assert len(pred[i])==len(targ[i])
            all+=len(pred[i])
            for j in range(len(pred[i])):
                if pred[i][j]==targ[i][j]:
                    acc+=1.0
        return acc/all
    def write_result(self,pred,targ):
        print('writing_result...........')
        f=open('./result/multi_0528_long_pred_lstm2.txt','w')
        f2=open('./result/multi_0528_long_targ_lstm2.txt','w')
        assert len(pred)==len(targ)
        for i in range(len(pred)):
            assert len(pred[i])==len(targ[i])
            f.write(' '.join(pred[i])+'\n')
            f2.write(' '.join(targ[i])+'\n')
        f.close()
        f2.close()
    def write_result2(self,sen,pred,targ):
        print('writing_result...........')
        f=open('./result/0602_long_pred_lstm.txt','w')
        assert len(pred)==len(targ)==len(sen)
        for i in range(len(pred)):
            for k in range(len(pred[i])):
                # print(len(pred[i][k]),len(targ[i]),len(sen[i]))
                assert len(pred[i][k])==len(targ[i])==len(sen[i])
                for j in range(len(targ[i])):
                    f.write(' '.join([sen[i][j],pred[i][k][j],targ[i][j]])+'\n')
                f.write('\n')
        f.close()


    def train(self,data, data_size):
        trans=self.trans_probability(data)
        if data_size is not None:
            total_steps = self.epochs * data_size
        else:
            total_steps = None

        self.model.train(True)
        global_step = 0
        global_iter = 0
        progress = tqdm.tqdm(total=total_steps)
        for e_idx in range(self.epochs):
            self.scheduler.step()
            # print(self.optimizer)
            # break
            loss=0.0
            preds_all=[]
            y_var_all=[]
            x_var_all=[]
            nll_v_all=0.0
            for i_idx, (batch, lens) in enumerate(data):
                self.model.zero_grad()
                xs, y = batch[:-1], batch[-1]
                xs_var, y_var, lens_s = self.prepare_batch(xs, y, lens)
                loglik, logits = self.model.loglik(xs_var, y_var, lens_s,trans, return_logits=True)
                '''
                nll = -loglik.mean()


                nll_v = float(-(loglik / lens_s.float()).data[0])
                '''
                nll_v=loglik
                
                nll_v.backward()
                nll_v_all+=nll_v.data.tolist()[0]
                self.optimizer.step()
                #break

                #预测结果
                preds = self.model.predict(xs_var, y_var,lens_s,trans)
                #print(preds[0])
            
                

                lens_s = lens_s.cpu().data.tolist()
                preds = preds.cpu().data.tolist()
                preds = self.tighten(preds, lens_s)
                preds_all.extend(preds)
                
                y_var = y_var.cpu().data.tolist()
                y_var = self.tighten(y_var, lens_s)
                y_var_all.extend(y_var)

                x_var = xs_var[0].cpu().data.tolist()
                x_var = self.tighten(x_var, lens_s)
                x_var_all.extend(x_var)

                

                batch_size = batch[0].size(0)
                global_step += batch_size
                global_iter += 1
                progress.set_description("nll={}".format(nll_v_all/(i_idx+1)))
                progress.update(batch_size)
                # break
            print('epoch====',e_idx)
                
             
            

            #print(type(self.input_vocabs),type(self.input_vocabs[0]))
            preds_all = [self.lexicalize_label(s, self.label_vocab) for s in preds_all]
            x_var_all = [self.lexicalize_data(s, self.input_vocabs[0]) for s in x_var_all]
            y_var_all = [self.lexicalize_label(s, self.label_vocab) for s in y_var_all]
            acc=self.compute_label_acc(preds_all,y_var_all)
            print(preds_all[-1-e_idx])
            print(y_var_all[-1-e_idx])
            print('acc=',acc) 
            torch.save(self.model,'./pkl/0604_multi_lstm'+str(e_idx)+'.pkl')
            # break
            
        #self.write_result(preds_all,y_var_all)
            #break

    def test(self,train_data,data, data_size):
        trans=self.trans_probability(train_data)
        print('begin testing ............')
        self.model.train(False)
        #loss=0.0
        preds_all=[]
        y_var_all=[]
        x_var_all=[]
        #nll_v_all=0.0
        for i_idx, (batch, lens) in enumerate(data):
            self.model.zero_grad()
            xs, y = batch[:-1], batch[-1]
            xs_var, y_var, lens_s = self.prepare_batch(xs, y, lens)
            # print(type(xs_var),type(lens_s))
            #loglik, logits = self.model.loglik(xs_var, y_var, lens_s, return_logits=True)
            # nll = -loglik.mean()

            # '''
            # nll_v = float(-(loglik / lens_s.float()).data[0])
            # nll_v_all+=nll_v
            # nll.backward()
            # self.optimizer.step()
            # '''

            #预测结果
            # preds = self.model.predict_for_bs(xs_var, lens_s,trans)
            preds = M.predict_out(self.model,xs_var, lens_s, trans)

            lens_s = lens_s.cpu().data.tolist()
            # print(lens_s[:10])
            #preds = preds.cpu().data.tolist()
            #preds = self.tighten(preds, lens_s)
            preds_all.extend(preds)
            
            y_var = y_var.cpu().data.tolist()
            y_var = self.tighten(y_var, lens_s)
            y_var_all.extend(y_var)
            print(y_var[-1])

            x_var = xs_var[0].cpu().data.tolist()
            x_var = self.tighten(x_var, lens_s)
            x_var_all.extend(x_var)
            break
                

        # print(len(preds_all),len(y_var_all))
        # new_preds=[]
        # for preds in preds_all:
        #     new_preds.append([self.lexicalize_label(s, self.label_vocab) for s in preds])

        # # preds_all = [self.lexicalize_label(s, self.label_vocab) for s in preds_all]
        # y_var_all = [self.lexicalize_label(s, self.label_vocab) for s in y_var_all]
        # x_var_all = [self.lexicalize_data(s, self.input_vocabs[0]) for s in x_var_all]
        # #acc=self.compute_label_acc(preds_all,y_var_all)
        # #print(preds_all[-1-e_idx])
        # #print(y_var_all[-1-e_idx])
        # #print('acc=',acc) 
        # self.write_result2(x_var_all,new_preds,y_var_all)
