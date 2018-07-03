import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
from torch.autograd import Variable
import numpy as np
import math
import copy


def log_sum_exp(vec, dim=0):
    max, idx = torch.max(vec, dim)
    max_exp = max.unsqueeze(-1).expand_as(vec)
    return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))




class LSTMCRF(nn.Module):
    def __init__(self, label_size, vocab_sizes, label_vocab,word_dims, hidden_dim, layers,
                 dropout_prob,flag, bidirectional=True):
        super(LSTMCRF, self).__init__()
        #++++++++++
        self.label_size = label_size
        self.n_labels = n_labels = label_size
        self.label_vocab=label_vocab
        #++++++++

        self.n_feats = len(word_dims)
        #print(sum(word_dims))
        self.total_word_dim = sum(word_dims)
        self.word_dims = word_dims
        self.hidden_dim = hidden_dim
        self.lstm_layers = layers
        self.dropout_prob = dropout_prob
        self.flag = flag
        self.is_cuda = False

        #self.crf = crf
        self.bidirectional = bidirectional
        #self.n_labels = n_labels = self.crf.n_labels
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, word_dim)
             for vocab_size, word_dim in zip(vocab_sizes, word_dims)]
        )

        self.output_hidden_dim = self.hidden_dim
        if bidirectional:
            self.output_hidden_dim *= 2

        self.tanh = nn.Tanh()
        self.softmax=torch.nn.Softmax()
        self.loss=torch.nn.CrossEntropyLoss()
        self.input_layer = nn.Linear(self.total_word_dim, hidden_dim)
        self.output_layer = nn.Linear(self.output_hidden_dim, n_labels)
        self.label_layer = nn.Linear(n_labels, n_labels)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=layers,
                            bidirectional=bidirectional,
                            dropout=dropout_prob,
                            batch_first=True)
        '''
        self.lstm2 = nn.LSTM(input_size=hidden_dim*2,
                            hidden_size=hidden_dim*2,
                            num_layers=layers,
                            bidirectional=False,
                            dropout=dropout_prob,
                            batch_first=True)
        '''

    def reset_parameters(self):
        for emb in self.embeddings:
            I.xavier_normal(emb.weight.data)

        I.xavier_normal(self.input_layer.weight.data)
        I.xavier_normal(self.output_layer.weight.data)
        I.xavier_normal(self.label_layer.weight.data)
        
        #self.crf.reset_parameters()
        self.lstm.reset_parameters()
        #self.lstm2.reset_parameters()

    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        x_packed = R.pack_padded_sequence(x, x_lens.data.tolist(),
                                          batch_first=True)

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=True)

        return output, h

    def _embeddings(self, xs):
        """Takes raw feature sequences and produces a single word embedding

        Arguments:
            xs: [n_feats, batch_size, seq_len] LongTensor

        Returns:
            [batch_size, seq_len, word_dim] FloatTensor 
        """
        n_feats, batch_size, seq_len = xs.size()

        assert n_feats == self.n_feats

        res = [emb(x) for emb, x in zip(self.embeddings, xs)]
        x = torch.cat(res, 2)

        return x
    def _make_label_array(self,y,lens,trans):
        # print(y.size())
        batch_size, seq_len = y.size()
        label_info=[]#torch.FloatTensor(batch_size,seq_len,self.n_labels).zero_()
        lenss=lens.data.numpy().tolist()
        ys=y.data.numpy().tolist()
        for i in range(batch_size):
            label_info.append(trans[self.flag])
            for j in range(seq_len-1):
                label_info.append(trans[ys[i][j]])
                # label_info[i][j][ys[i][j]]=1.0
        label_info=torch.FloatTensor(label_info)
        # print('label_info==',label_info.size())
        return label_info



    def _forward_bilstm(self, xs,y, lens,trans):
        # print('y==',y.size())
        label_info=Variable(self._make_label_array(y,lens,trans),requires_grad=False)
        # print('label_info==',label_info.size())
        n_feats, batch_size, seq_len = xs.size()

        x = self._embeddings(xs)
        x = x.view(-1, self.total_word_dim)
        x = self.tanh(self.input_layer(x))
        x = x.view(batch_size, seq_len, self.hidden_dim)

        o, h = self._run_rnn_packed(self.lstm, x, lens)
        #print('o====',o.size())
        #o, h = self._run_rnn_packed(self.lstm2, o, lens)

        o = o.contiguous()
        o = o.view(-1, self.output_hidden_dim)
        o = self.tanh(self.output_layer(o))
        label_info=label_info.view(-1, self.n_labels)
        label_info=self.tanh(self.label_layer(label_info))
        o = self.softmax(o*label_info)###################
        o = o.view(batch_size, seq_len, self.n_labels)
        #print(o[0])

        return o
    

    def mask_bios(self,xs,y):
        #print('flag==',self.flag)
        seq_len,label_size=xs.size()
        bios=torch.FloatTensor(seq_len,label_size).zero_()
        y=y.data.numpy().tolist()
        for i in range(len(y)):
            if y[i]==self.flag:
                bios[i][y[i]]=1.0
            else:
                bios[i][y[i]]=2.0
        return bios


    def myloss(self, logits, y, lens):
        #print(logits[0])
        batch_size,seq_len,label_size=logits.size()
        logits=logits.view(-1,label_size)
        #print('myloss')
        y_exp = y.unsqueeze(-1)
        bios=Variable(self.mask_bios(logits,y).float(),requires_grad=False)
        #mask = sequence_mask(lens).float()
        #print('mask==',mask.size(),type(mask))
        scores=-torch.log(logits)*bios
        scores=scores.view(batch_size,seq_len,label_size)
        #scores=scores*mask
        #print(scores.size())
        #print(scores[0])
        scores=scores.view(-1,seq_len*label_size)
        #scores= torch.max(scores,1)[0]
        scores=torch.sum(scores)
        return scores

    

    def _bilstm_score(self, logits, y, lens):
        y_exp = y.unsqueeze(-1)
        batch_size,seq_len,label_size=logits.size()
        
        logits=logits.view(-1,label_size)
        y=y.view(batch_size*seq_len)
        #scores = torch.gather(logits, 2, y_exp).squeeze(-1)
        # scores = self.loss(logits,y)
        logits=logits.view(batch_size,seq_len,label_size)
        scores = self.myloss(logits,y,lens)

        return scores

    def score(self, xs, y, lens, logits=None):
        if logits is None:
            logits = self._forward_bilstm(xs, y, lens)

        #transition_score = self.crf.transition_score(y, lens)
        bilstm_score = self._bilstm_score(logits, y, lens)

        score = bilstm_score   #+transition_score 

        return score
    #在测试过程中返回batch中一个句子的结果，未经过softmax
    def _forward_bilstm_for_test(self, xs, lens, i):
        # print("former_label===",former_label)
        # label_list=[0.0]*self.n_labels
        # label_list[former_label]=1.0
        # label_list=torch.FloatTensor(label_list)
        #print(label_list)
        n_feats, batch_size, seq_len = xs.size()

        x = self._embeddings(xs)
        # print('x====',x.size())
        x = x[i]#i时刻
        x = x.view(-1, self.total_word_dim)
        x = self.tanh(self.input_layer(x))
        x = x.view(-1, seq_len, self.hidden_dim)

        o, h = self._run_rnn_packed(self.lstm, x, lens[i])
        #print('o====',o.size())
        #o, h = self._run_rnn_packed(self.lstm2, o, lens)

        o = o.contiguous()
        o = o.view(-1, self.output_hidden_dim)
        o = self.tanh(self.output_layer(o))
        # print('o===',o.size())

        # o = self.softmax(o)
        # seq_len=lens[i].data.numpy().tolist()[0]
        # o = o.view(-1, seq_len, self.n_labels)
        #print(o[0])

        return o
    def beam_search(self,output,j,k,former_label,trans):
        #print("former_label===",former_label)
        # label_list=[0.0]*self.n_labels
        # label_list[former_label]=1.0
        label_list=trans[former_label]
        label_list=Variable(torch.FloatTensor(label_list),requires_grad=False)
        label_list=label_list.view(-1, self.n_labels)
        label_list=self.tanh(self.label_layer(label_list))
        o=output[j]+label_list.contiguous()
        o = self.softmax(o)
        o=o.view(self.n_labels)
        sort, indices = torch.sort(o,descending=True)
        return sort[:k],indices[:k]
    def extend_res(self,res,top_k_p,top_k,sen_len,former_label):
        top_k=top_k.data.numpy().tolist()
        top_k_p=top_k_p.data.numpy().tolist()
        if res==[]:
            for k in range(len(top_k)):
                res.append([[top_k[k]],math.log(top_k_p[k])])
            return res

        # remove_list=[]
        # for r in res:
        #     if len(r[0])==sen_len and r[0][-1]==former_label:
        #         remove_list.append(r)
                
        #         for k in range(len(top_k)):
        #             temp=[[],r[1]]
        #             for ri in r[0]:
        #                 temp[0].append(ri)

        #             # temp=copy.copy(r)
        #             temp[0].append(top_k[k])
        #             temp[1]+=math.log(top_k_p[k])
        #             res.append(temp)
        # # print('len(res)===',len(res))
        # for r in remove_list:
        #     res.remove(r)
        # # print('len(res)===',len(res))
        return res
    def extend_res_signal(self,r,top_k_p,top_k):
        top_k=top_k.data.numpy().tolist()
        top_k_p=top_k_p.data.numpy().tolist()
        res=[]
        for k in range(len(top_k)):
            temp=[[],r[1]]
            for ri in r[0]:
                temp[0].append(ri)

            # temp=copy.copy(r)
            temp[0].append(top_k[k])
            temp[1]+=math.log(top_k_p[k])
            res.append(temp)
        return res

    # def get_former_label(self,res):
    #     #print('res===',res)
    #     if res==[]:
    #         return [self.label_vocab.f2i['O']]
    #     else:
    #         label=[]
    #         for i in range(len(res)):
    #             l=res[i][0][-1]
    #             if l not in label:
    #                 label.append(l)
    #         return label

    def predict_for_bs(self, xs, lens, trans,return_scores=False,k=5):
        # former_label=self.label_vocab.f2i["START"]
        # print('k===',k)
        lenss=lens.data.numpy().tolist()
        result_all=[]
        # print(xs.size(),xs.size(1))
        for j in range(1):#xs.size(1)
            i=len(lenss)-1
            res_temp=[]#最终的k个结果
            output = self._forward_bilstm_for_test(xs, lens,i)
            res=[]#暂存所有可能的结果
            while(len(res_temp)<k):
                if res==[]:
                    former_label=self.label_vocab.f2i['O']
                    top_k_p,top_k=self.beam_search(output,0,self.n_labels,former_label,trans)
                    res=self.extend_res(res,top_k_p,top_k,0,former_label)
                    # print('len_res==',len(res))
                else:
                    #每一步扩展k个候选
                    res.sort(key=lambda k:k[1])
                    len_all=len(res)
                    if len(res)>1000:
                        res=res[-int(len_all/2):]
                    len_all=len(res)
                    res1=res[-k:]#top_k
                    res=res[:len_all-k]#the other
                    # print('len_res3==',len(res))
                    for r in res1:
                        #对达到句子长度的候选不做扩展
                        if len(r[0])>=lenss[i]:
                            continue
                        former_label=r[0][-1]
                        print('len_r==',len(r[0]),r[1])
                        top_k_p,top_k=self.beam_search(output,len(r[0]),k,former_label,trans)
                        temp=self.extend_res_signal(r,top_k_p,top_k)
                        res.extend(temp)
                    del res1
                    
                    #扩展后的得分最高的句子达到句子长度，则加入到res_temp中
                    res.sort(key=lambda k:k[1])
                    temp=res[-1]
                    if len(temp[0])==lenss[i]:
                        print('yes======',len(res_temp)+1)
                        res_temp.append(temp)
                        res.remove(temp)

                # print('lenss===',lenss[0],len(res[0][0]))
                
            result=[]
            for i in range(len(res_temp)):
                result.append(res_temp[i][0])
                print(res_temp[i][0])
            result_all.append(result)
                
            
        # sizes=logits.size()
        # logits = logits.view(-1,self.n_labels)
        # res=torch.max(logits,1)[1]
        # res=res.view(sizes[0],sizes[1])
        # return res
        # print(result_all[-1])
        return result_all

    def predict(self, xs, y,lens, trans,return_scores=False):
        logits = self._forward_bilstm(xs, y,lens,trans)
        #print("logits==",logits.size())
        sizes=logits.size()
        logits = logits.view(-1,self.n_labels)
        res=torch.max(logits,1)[1]
        res=res.view(sizes[0],sizes[1])
        return res
        '''
        scores, preds = self.crf.viterbi_decode(logits, lens)

        if return_scores:
            return preds, scores
        else:
            return preds
    #     '''

    def loglik(self, xs, y, lens,trans, return_logits=False):
        #print(xs.size())
        #print(lens)
        #print('loglik')
        logits = self._forward_bilstm(xs, y,lens,trans)
        #norm_score = self.crf(logits, lens)
        sequence_score = self.score(xs, y, lens, logits=logits)
        loglik = sequence_score #- norm_score

        if return_logits:
            return loglik, logits
        else:
            return loglik





def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = Variable(ranges)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask
def extend_res_signal(r,top_k_p,top_k):
    top_k=top_k.data.numpy().tolist()
    top_k_p=top_k_p.data.numpy().tolist()
    res=[]
    for k in range(len(top_k)):
        temp=[[],r[1]]
        for ri in r[0]:
            temp[0].append(ri)

        # temp=copy.copy(r)
        temp[0].append(top_k[k])
        temp[1]+=math.log(top_k_p[k])
        res.append(temp)
    return res

def predict_out(model,xs, lens, trans,k=40):
    # print(type(xs))
    lenss=lens.data.numpy().tolist()
    result_all=[]
    # print(xs.size(),xs.size(1))
    for j in range(1):#xs.size(1)
        i=len(lenss)-1
        res_temp=[]#最终的k个结果
        output = model._forward_bilstm_for_test(xs, lens,i)
        res=[]#暂存所有可能的结果
        while(len(res_temp)<k):
            if res==[]:
                former_label=model.label_vocab.f2i['O']
                top_k_p,top_k=model.beam_search(output,0,model.n_labels,former_label,trans)
                res=model.extend_res(res,top_k_p,top_k,0,former_label)
                # print('len_res==',len(res))
            else:
                #每一步扩展k个候选
                res.sort(key=lambda k:k[1])
                len_all=len(res)
                if len(res)>10000:
                    res=res[-int(len_all/2):]
                len_all=len(res)
                res1=res[-k:]#top_k
                res=res[:len_all-k]#the other
                # print('len_res3==',len(res))
                for r in res1:
                    #对达到句子长度的候选不做扩展
                    if len(r[0])>=lenss[i]:
                        continue
                    former_label=r[0][-1]
                    # print('len_r==',len(r[0]),r[1])
                    top_k_p,top_k=model.beam_search(output,len(r[0]),k,former_label,trans)
                    temp=extend_res_signal(r,top_k_p,top_k)
                    res.extend(temp)
                del res1
                
                #扩展后的得分最高的句子达到句子长度，则加入到res_temp中
                res.sort(key=lambda k:k[1],reverse=False)
                # print('0=====',res[0][1])
                temp=res[-1]
                if len(temp[0])==lenss[i]:
                    print('-1=====',temp[1])
                    print('yes======',len(res_temp)+1)
                    res_temp.append(temp)
                    res.remove(temp)

            # print('lenss===',lenss[0],len(res[0][0]))
            
        result=[]
        for i in range(len(res_temp)):
            result.append(res_temp[i][0])
            print(res_temp[i][0])
        result_all.append(result)
    return result_all

