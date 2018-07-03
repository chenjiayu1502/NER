import copy

import torch
import numpy as np
import torch.utils.data as D

import utils as U


class MultiSentWordDataset(D.Dataset):
    def __init__(self, *paths):
        self.data = [
            list(U.FileReader(path).sents()) for path in paths
        ]
        assert all(len(d) == len(self.data[0]) for d in self.data), \
            "Not all files have the same length."

    def __getitem__(self, idx):
        return tuple(data[idx] for data in self.data)

    def __len__(self):
        return len(self.data[0])

    def split(self, *ratios, shuffle=False):
        idx = np.arange(len(self))

        if shuffle:
            idx = np.random.permutation(idx)

        ratios = [r / sum(ratios) for r in ratios]
        counts = [int(round(len(self) * r)) for r in ratios]
        cum_counts = [sum(counts[:i + 1]) for i in range(len(ratios))]
        bounds = [0] + cum_counts

        for i in range(len(bounds) - 1):
            s = copy.copy(self)
            s.data = [[data[j] for j in idx[bounds[i]:bounds[i + 1]]]
                      for data in self.data]

            yield s


class MultiSentWordDataLoader(D.DataLoader):
    def __init__(self, dataset, input_vocabs, label_vocabs, tensor_lens=True, **kwargs):
        super(MultiSentWordDataLoader, self).__init__(
            dataset=dataset,
            collate_fn=self.collate_fn,
            **kwargs
        )

        self.tensor_lens = tensor_lens
        self.input_vocabs = input_vocabs[0]

        self.label_vocabs = label_vocabs

    def unlexicalize_data(self, sent, vocab):
        return [vocab.f2i[w] if w in vocab else vocab.f2i["<unk>"]
                for w in sent]

    def unlexicalize_label(self, sent, vocab):
        return [vocab.f2i[w] if w in vocab else vocab.f2i["O"]
                for w in sent]

    def pad(self, sent, max_len, pad_idx):
        if len(sent) >= max_len:
            sent = sent[:max_len]
            return sent

        return sent + [pad_idx] * (max_len - len(sent))

    def collate_fn(self, batches):
        sents_list = list(zip(*batches))

        ret = []
        lens = [len(s) for s in sents_list[0]]
        max_len = max(lens)

        sents = [self.unlexicalize_data(s, self.input_vocabs) for s in sents_list[0]]
        sents = [self.pad(sent, max_len, self.input_vocabs.f2i["<pad>"])for sent in sents]
        sents = torch.LongTensor(sents)
        ret.append(sents.unsqueeze(0))

        sents = [self.unlexicalize_label(s, self.label_vocabs) for s in sents_list[1]]
        sents = [self.pad(sent, max_len, self.label_vocabs.f2i["O"])for sent in sents]
        sents = torch.LongTensor(sents)
        ret.append(sents.unsqueeze(0))

        # for sents, vocab in zip(sents_list, self.vocabs):
        #     sents = [self.unlexicalize_data(s, vocab) for s in sents]
        #     sents = [self.pad(sent, max_len, vocab.f2i["<pad>"])
        #              for sent in sents]
        #     sents = torch.LongTensor(sents)
        #     ret.append(sents.unsqueeze(0))

        lens = torch.LongTensor(lens)
        return torch.cat(ret), lens
    
