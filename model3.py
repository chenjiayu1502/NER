import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
from torch.autograd import Variable


def log_sum_exp(vec, dim=0):
    max, idx = torch.max(vec, dim)
    max_exp = max.unsqueeze(-1).expand_as(vec)
    return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))




class LSTMCRF(nn.Module):
    def __init__(self, label_size, vocab_sizes, word_dims, hidden_dim, layers,
                 dropout_prob, bidirectional=True):
        super(LSTMCRF, self).__init__()
        #++++++++++
        self.label_size = label_size
        self.n_labels = n_labels = label_size
        #++++++++

        self.n_feats = len(word_dims)
        #print(sum(word_dims))
        self.total_word_dim = sum(word_dims)
        self.word_dims = word_dims
        self.hidden_dim = hidden_dim
        self.lstm_layers = layers
        self.dropout_prob = dropout_prob
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
        self.softmax=torch.nn.LogSoftmax()
        self.loss=torch.nn.CrossEntropyLoss()
        self.input_layer = nn.Linear(self.total_word_dim, hidden_dim)
        self.output_layer = nn.Linear(self.output_hidden_dim, n_labels)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=layers,
                            bidirectional=bidirectional,
                            dropout=dropout_prob,
                            batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim*2,
                            hidden_size=hidden_dim*2,
                            num_layers=layers,
                            bidirectional=False,
                            dropout=dropout_prob,
                            batch_first=True)

    def reset_parameters(self):
        for emb in self.embeddings:
            I.xavier_normal(emb.weight.data)

        I.xavier_normal(self.input_layer.weight.data)
        I.xavier_normal(self.output_layer.weight.data)
        #self.crf.reset_parameters()
        self.lstm.reset_parameters()

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

    def _forward_bilstm(self, xs, lens):
        n_feats, batch_size, seq_len = xs.size()

        x = self._embeddings(xs)
        x = x.view(-1, self.total_word_dim)
        x = self.tanh(self.input_layer(x))
        x = x.view(batch_size, seq_len, self.hidden_dim)

        o, h = self._run_rnn_packed(self.lstm, x, lens)
        #print('o====',o.size())
        o, h = self._run_rnn_packed(self.lstm2, o, lens)

        o = o.contiguous()
        o = o.view(-1, self.output_hidden_dim)
        o = self.tanh(self.output_layer(o))
        o = self.softmax(o)
        o = o.view(batch_size, seq_len, self.n_labels)

        return o

    def _bilstm_score(self, logits, y, lens):
        y_exp = y.unsqueeze(-1)
        batch_size,seq_len,label_size=logits.size()
        
        logits=logits.view(-1,label_size)
        y=y.view(batch_size*seq_len)
        #scores = torch.gather(logits, 2, y_exp).squeeze(-1)
        scores = self.loss(logits,y)

        return scores

    def score(self, xs, y, lens, logits=None):
        if logits is None:
            logits = self._forward_bilstm(xs, lens)

        #transition_score = self.crf.transition_score(y, lens)
        bilstm_score = self._bilstm_score(logits, y, lens)

        score = bilstm_score   #+transition_score 

        return score

    def predict(self, xs, lens, return_scores=False):
        logits = self._forward_bilstm(xs, lens)
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
        '''

    def loglik(self, xs, y, lens, return_logits=False):
        #print(xs.size())
        #print(lens)
        logits = self._forward_bilstm(xs, lens)
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


