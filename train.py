import os
import pickle
import shutil
import logging
import argparse
import tempfile
import subprocess
import collections

import numpy as np
import yaap
import tqdm
import torch
import torch.nn as nn
import torch.optim as O
import torch.autograd as A
import sys

import utils
import data as D
import model as M
import evaluate as E

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# test_data_path = "/home/hzq/kbqa/bi_lstm_crf/bilstm_crf_test_data.txt"
# test_label_path = "/home/hzq/kbqa/bi_lstm_crf/bilstm_crf_test_label.txt"


parser = yaap.ArgParser(
    allow_config=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

group = parser.add_group("Basic Options")
group.add("--input-path", type=yaap.path, action="append", required=True,
          help="Path to input file that contains sequences of tokens "
               "separated by spaces.")
group.add("--label-path", type=yaap.path, required=True,
          help="Path to label file that contains sequences of token "
               "labels separated by spaces. Note that the number of "
               "tokens in each sequence must be equal to that of the "
               "corresponding input sequence.")
group.add("--test-input-path", type=yaap.path, action="append", required=True,
          help="Path to input file that contains sequences of tokens "
               "separated by spaces.")
group.add("--test-label-path", type=yaap.path, required=True,
          help="Path to label file that contains sequences of token "
               "labels separated by spaces. Note that the number of "
               "tokens in each sequence must be equal to that of the "
               "corresponding input sequence.")
group.add("--save-dir", type=yaap.path, required=True,
          help="Directory to save outputs (checkpoints, vocabs, etc.)")
group.add("--gpu", type=int, action="append",
          help="Device id of gpu to use. Could supply multiple gpu ids "
               "to denote multi-gpu utilization. If no gpus are "
               "specified, cpu is used as default.")


group = parser.add_group("Training Options")
group.add("--epochs", type=int, default=1,
          help="Number of training epochs.")
group.add("--dropout-prob", type=float, default=0.5,
          help="Probability in dropout layers.")
group.add("--batch-size", type=int, default=128,
          help="Mini-batch size.")
group.add("--shuffle", action="store_true", default=False,
          help="Whether to shuffle the dataset.")
group.add("--ckpt-period", type=utils.PeriodChecker, default="1e",
          help="Period to wait until a model checkpoint is "
               "saved to the disk. "
               "Periods are specified by an integer and a unit ('e': "
               "epoch, 'i': iteration, 's': global step).")

group = parser.add_group("Validation Options")
group.add("--val", action="store_true", default=True,
          help="Whether to perform validation.")
group.add("--val-ratio", type=float, default=0.1)
group.add("--val-period", type=utils.PeriodChecker, default="100i",
          help="Period to wait until a validation is performed. "
               "Periods are specified by an integer and a unit ('e': "
               "epoch, 'i': iteration, 's': global step).")
group.add("--samples", type=int, default=1,
          help="Number of output samples to display at each iteration.")

group = parser.add_group("Model Parameters")
group.add("--word-dim", type=int, action="append",
          help="Dimensions of word embeddings. Must be specified for each "
               "input. Defaults to 300 if none is specified.")
group.add("--lstm-dim", type=int, default=300,
          help="Dimensions of lstm cells. This determines the hidden "
               "state and cell state sizes.")
group.add("--lstm-layers", type=int, default=1,
          help="Layers of lstm cells.")
group.add("--bidirectional", action="store_true", default=True,
          help="Whether lstm cells are bidirectional.")



class BaseLSTMCRFTrainer(object):
    def __init__(self, model: M.LSTMCRF, epochs=5, optimizer=O.Adam, gpus=None):
        self.model = model
        self.epochs = epochs
        self.optimizer_cls = optimizer
        self.optimizer = optimizer(self.model.parameters())
        self.gpus = gpus
        self.gpu_main = None

        if gpus is not None:
            self.gpus = self._ensure_tuple(self.gpus)
            self.gpu_main = self.gpus[0]
            self.model = M.TransparentDataParallel(
                self.model,
                device_ids=self.gpus,
                output_device=self.gpu_main
            )

    def wrap_var(self, x, **kwargs):
        x = A.Variable(x, **kwargs)

        if self.gpus is not None:
            x = x.cuda(self.gpu_main)

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


class LSTMCRFTrainer(BaseLSTMCRFTrainer):
    def __init__(self, sargs, input_vocabs, label_vocab, *args,
                 val_data=None, **kwargs):
        super(LSTMCRFTrainer, self).__init__(*args, **kwargs)

        self.args = sargs
        self.input_vocabs = input_vocabs
        self.label_vocab = label_vocab
        self.val_data = val_data
        self.writer = None

        self.repeatables = {
            self.args.ckpt_period: self.save_checkpoint
        }

        if self.args.val:
            self.repeatables[self.args.val_period] = \
                self.validate

    @staticmethod
    def get_longest_word(vocab):
        lens = [(len(w), w) for w in vocab.f2i]
        return max(lens, key=lambda x: x[0])[1]

    def display_row(self, items, widths):
        assert len(items) == len(widths)

        padded = ["{{:>{}s}}".format(w).format(item)
                  for item, w in zip(items, widths)]
        logging.info(" ".join(padded))

    def display_samples(self, inputs, targets, lstms, crfs):
        assert len(self.input_vocabs) == len(inputs), \
            "Number of input features do not match."

        inputs = [[self.lexicalize_data(s, v) for s in input]
                  for input, v in zip(inputs, self.input_vocabs)]
        targets = [self.lexicalize_label(s, self.label_vocab) for s in targets]
        lstms = [self.lexicalize_label(s, self.label_vocab) for s in lstms]
        crfs = [self.lexicalize_label(s, self.label_vocab) for s in crfs]
        transposed = list(zip(*(inputs + [lstms, crfs, targets])))
        col_names = ["INPUT{}".format(i + 1) for i in range(len(inputs))] + \
                    ["LSTM", "CRF", "TARGETS"]
        vocabs = self.input_vocabs + [self.label_vocab] * 3
        col_widths = [max(len(self.get_longest_word(v)), len(c))
                      for v, c in zip(vocabs, col_names)]

        for i, sample in enumerate(transposed):
            rows = list(zip(*sample))

            logging.info("")
            logging.info("SAMPLE #{}".format(i + 1))
            self.display_row(col_names, col_widths)
            for row in rows:
                self.display_row(row, col_widths)
    def write_data_res(self, inputs, targets, lstms, crfs):
        assert len(self.input_vocabs) == len(inputs), \
            "Number of input features do not match."

        inputs = [[self.lexicalize_data(s, v) for s in input]
                  for input, v in zip(inputs, self.input_vocabs)]
        targets = [self.lexicalize_label(s, self.label_vocab) for s in targets]
        print(targets)
        lstms = [self.lexicalize_label(s, self.label_vocab) for s in lstms]
        crfs = [self.lexicalize_label(s, self.label_vocab) for s in crfs]
        transposed = list(zip(*(inputs + [lstms, crfs, targets])))
        col_names = ["INPUT{}".format(i + 1) for i in range(len(inputs))] + \
                    ["LSTM", "CRF", "TARGETS"]
        vocabs = self.input_vocabs + [self.label_vocab] * 3
        col_widths = [max(len(self.get_longest_word(v)), len(c))
                      for v, c in zip(vocabs, col_names)]

        for i, sample in enumerate(transposed):
            rows = list(zip(*sample))

            logging.info("")
            logging.info("SAMPLE #{}".format(i + 1))
            self.display_row(col_names, col_widths)
            for row in rows:
                self.display_row(row, col_widths)

    def lexicalize_data(self, seq, vocab):
        return [vocab.i2f[w] if w in vocab else "<unk>" for w in seq]

    def lexicalize_label(self, seq, vocab):
        return [vocab.i2f[w] if w in vocab else 'O' for w in seq]

    @staticmethod
    def tighten(seqs, lens):
        return [s[:l] for s, l in zip(seqs, lens)]

    @staticmethod
    def random_idx(max_count, subset=None):
        idx = np.random.permutation(np.arange(max_count))

        if subset is not None:
            return idx[:subset]

        return idx

    @staticmethod
    def gather(lst, idx):
        return [lst[i] for i in idx]

    def validate(self, epochs=None, iters=None, steps=None):
        if not self.args.val:
            return

        logging.info("Validating...")
        self.model.train(False)
        nll_all = 0
        lstm_all, preds_all, targets_all = [], [], []
        data_size = 0
        sampled = False

        for i_idx, (batch, lens) in enumerate(self.val_data):
            i_idx += 1
            batch_size = batch.size(1)
            data_size += batch_size

            xs, y = batch[:-1], batch[-1]
            xs_var, y_var, lens_s = self.prepare_batch(xs, y, lens,
                                                       volatile=True)
            loglik, logits = self.model.loglik(xs_var, y_var, lens_s,
                                               return_logits=True)
            nll = -loglik.mean()
            nll_v = float(-(loglik / lens_s.float()).data[0])

            preds = self.model.predict(xs_var, lens_s)
            preds = preds.cpu().data.tolist()
            targets = y_var.cpu().data.tolist()
            lens_s = lens_s.cpu().data.tolist()

            preds = self.tighten(preds, lens_s)
            targets = self.tighten(targets, lens_s)

            preds_all.extend(preds)
            targets_all.extend(targets)
            nll_all = nll_v * batch_size

            lstm = logits.max(2)[1]
            lstm = lstm.cpu().data.tolist()
            lstm = self.tighten(lstm, lens_s)
            lstm_all.extend(lstm)

            if not sampled and self.args.samples > 0:
                sample_idx = self.random_idx(batch_size, self.args.samples)
                '''
                print('sample_idx===',type(sample_idx),sample_idx)
                num=[i for i in range(batch_size)]
                sample_idx_list=np.array(num)
                sample_idx=sample_idx_list[0]
                print('sample_idx===',type(sample_idx),sample_idx)
                '''
                xs = xs_var.cpu().data.tolist()
                xs = [self.tighten(x, lens_s) for x in xs]

                xs_smp = [self.gather(x, sample_idx) for x in xs]
                y_smp = self.gather(targets, sample_idx)
                crf_smp = self.gather(preds, sample_idx)
                lstm_smp = self.gather(lstm, sample_idx)

                #self.display_samples(xs_smp, y_smp, lstm_smp, crf_smp)
                self.write_data_res(xs_smp, y_smp, lstm_smp, crf_smp)
                del xs, lstm, xs_smp, y_smp, crf_smp, lstm_smp

                sampled = True

        nll = nll_all / data_size
        preds_all = [self.lexicalize_label(s, self.label_vocab) for s in preds_all]
        targets_all = [self.lexicalize_label(s, self.label_vocab)
                       for s in targets_all]
        lstm_all = [self.lexicalize_label(s, self.label_vocab) for s in lstm_all]

        preds_all = E.preprocess_labels(preds_all)
        targets_all = E.preprocess_labels(targets_all)
        lstm_all = E.preprocess_labels(lstm_all)

        lprec, lrec, lf1 = E.compute_f1(lstm_all, targets_all)
        prec, rec, f1 = E.compute_f1(preds_all, targets_all)
        print("nll:", nll, "lstm_prec:", lprec, "lstm_lrec:", lrec, "lstm_lf1:", lf1)
        print("nll:", nll, "lstmcef_prec:", prec, "lstmcef_:", rec, "lstmcef_f1:", f1)

        del preds_all, targets_all, lstm_all

    def train(self, data, data_size=None):
        if data_size is not None:
            total_steps = self.epochs * data_size
        else:
            total_steps = None

        self.model.train(True)
        global_step = 0
        global_iter = 0
        progress = tqdm.tqdm(total=total_steps)
        i_step = 0
        plot_x_iteration = []
        plot_y_loss = []
        lp, lr, lf1 = [], [], []
        p, r, f1 = [], [], []
        for e_idx in range(self.epochs):
            e_idx += 1
            for i_idx, (batch, lens) in enumerate(data):
                '''
                print('i_idx=',i_idx)
                print('batch=',batch)
                print('lens=',lens)
                '''
                i_step += 1
                batch_size = batch[0].size(0)
                i_idx += 1
                global_step += batch_size
                global_iter += 1
                self.model.zero_grad()

                xs, y = batch[:-1], batch[-1]
                xs_var, y_var, lens_s = self.prepare_batch(xs, y, lens)

                loglik, logits = self.model.loglik(xs_var, y_var, lens_s, return_logits=True)
                nll = -loglik.mean()
                nll_v = float(-(loglik / lens_s.float()).data[0])
                nll.backward()
                self.optimizer.step()

                plot_x_iteration.append(i_step)
                plot_y_loss.append(nll_v)

                progress.set_description("nll={}".format(nll_v))
                progress.update(batch_size)
                break

            plt.figure()
            # plt.plot(plot_x_iteration, lp, 'g-', label='LSTM_Precision')
            # plt.plot(plot_x_iteration, p, 'b-', label='LSTM_CRF_Precision')
            plt.plot(plot_x_iteration, plot_y_loss, 'r-', label='Loss')
            plt.legend()
            plt.xlabel("Iteration")
            self.validate()
            break
        plt.savefig("/home/hzq/cjy/loss.1.jpg")
        del lf1, lp, lr, f1, p, r

    def test(self, data, data_size=None):
        f_test=open('test_result.txt','w')
        f_targ=open('target_result.txt','w')
        if data_size is not None:
            total_steps = data_size
        else:
            total_steps = None

        self.model.train(False)
        global_step = 0
        global_iter = 0
        progress = tqdm.tqdm(total=total_steps)
        i_step = 0

        ll, lp, lt = [], [], []
        lf1 = []
        f1 = []
        x_step = []
        for i_idx, (batch, lens) in enumerate(data):
            # print("batch:", batch, "\nlens:", lens)
            i_step += 1
            batch_size = batch[0].size(0)
            i_idx += 1
            global_step += batch_size
            global_iter += 1
            self.model.zero_grad()

            xs, y = batch[:-1], batch[-1]
            xs_var, y_var, lens_s = self.prepare_batch(xs, y, lens)

            loglik, logits = self.model.loglik(xs_var, y_var, lens_s, return_logits=True)
            nll = -loglik.mean()
            nll_v = float(-(loglik / lens_s.float()).data[0])

            preds = self.model.predict(xs_var, lens_s)
            preds = preds.cpu().data.tolist()
            targets = y_var.cpu().data.tolist()
            lens_s = lens_s.cpu().data.tolist()
            

            lstm = logits.max(2)[1]
            lstm = lstm.cpu().data.tolist()
            lstm = self.tighten(lstm, lens_s)

            lstm = [self.lexicalize_label(s, self.label_vocab) for s in lstm]
            preds = [self.lexicalize_label(s, self.label_vocab) for s in preds]
            targets = [self.lexicalize_label(s, self.label_vocab) for s in targets]
            lstm = E.preprocess_labels(lstm)
            preds = E.preprocess_labels(preds)
            targets = E.preprocess_labels(targets)
            

            preds = self.tighten(preds, lens_s)
            targets = self.tighten(targets, lens_s)
            

            ll.extend(lstm)
            lp.extend(preds)
            lt.extend(targets)
            for line in lp:
                temp=[str(wk) for wk in line]
                f_test.write(' '.join(temp)+'\n')
            for line in lt:
                temp=[str(wk) for wk in line]
                f_targ.write(' '.join(temp)+'\n')

            progress.update(batch_size)
            progress.set_description("nll={}".format(nll_v))

            lp_, lr_, lf1_ = E.compute_f1(ll, lt)
            p_, r_, f1_ = E.compute_f1(lp, lt)
            lf1.append(lf1_)
            f1.append(f1_)
            x_step.append(i_idx)
            break

        plt.figure()
        plt.plot(x_step, lf1, 'r-', label='LSTM_F1')
        plt.plot(x_step, f1, 'b-', label='LSTMCRF_F1')
        plt.legend()
        plt.xlabel("Iteration")
        plt.savefig("/home/hzq/cjy/f1.jpg")

    def save_checkpoint(self, epochs=None, iters=None, steps=None):
        logging.info("Saving checkpoint...")
        if isinstance(self.model, nn.DataParallel):
            module = self.model.module
        else:
            module = self.model
        state_dict = module.state_dict()

        if epochs is not None:
            name = "ckpt-e{}".format(epochs)
        elif iters is not None:
            name = "ckpt-i{}".format(iters)
        else:
            name = "ckpt-s{}".format(steps)

        save_path = os.path.join(args.save_dir, name)
        torch.save(state_dict, save_path)
        logging.info("Checkpoint saved to '{save_path}'.")

    def on_iter_complete(self, loss, local_iter, global_iter, global_step):
        for period_checker, func in self.repeatables.items():
            if period_checker(iters=global_iter, steps=global_step):
                func(iters=global_iter, steps=global_step)

    def on_epoch_complete(self, epoch_idx, global_iter, global_step):
        for period_checker, func in self.repeatables.items():
            if period_checker(epochs=epoch_idx,
                              iters=global_iter,
                              steps=global_step):
                func(epochs=epoch_idx,
                     iters=global_iter,
                     steps=global_step)


def check_arguments(args):
    num_inputs = len(args.input_path)

    assert num_inputs > 0, \
        "At least one input file must be specified."

    defaults = {
        # "wordembed-type": "none",
        # "wordembed-path": None,
        # "wordembed-freeze": False,
        "word-dim": 300,
    }

    # default values for list type arguments
    for attr, default in defaults.items():
        attr_name = attr.replace("-", "_")
        if getattr(args, attr_name) is None:
            setattr(args, attr_name, [default] * num_inputs)

    # check if input counts are correct
    for attr in defaults:
        attr_name = attr.replace("-", "_")
        assert len(getattr(args, attr_name)) == num_inputs, \
            "--{} must be specified as many as inputs.\n specified: {}, required: {}".format(attr,
                                                                                             len(getattr(args,
                                                                                                         attr_name)),
                                                                                             num_inputs)

    assert 0.0 < args.val_ratio < 1.0, \
        "Specify a valid validation ratio."

    # ensure that the save-dir exists
    os.makedirs(args.save_dir, exist_ok=True)


def main(args):
    logging.basicConfig(level=logging.INFO)
    check_arguments(args)

    logging.info("Creating vocabulary...")
    input_vocabs = []

    for input in args.input_path:
        vocab = utils.Vocabulary()
        words = utils.FileReader(input).words()
        vocab.add("<pad>")
        vocab.add("<unk>")
        utils.populate_vocab(words, vocab)
        input_vocabs.append(vocab)
    # print(input_vocabs)[<utils.Vocabulary object at 0x7fa839f5a0b8>]

    label_vocab = utils.Vocabulary()
    words = utils.FileReader(args.label_path).words()
    label_vocab.add("START")
    label_vocab.add("END")
    utils.populate_vocab(words, label_vocab)

    for i, input_vocab in enumerate(input_vocabs):
        vocab_path = os.path.join(args.save_dir,
                                  "vocab-input{}.pkl".format(i + 1))
        pickle.dump(input_vocab, open(vocab_path, "wb"))
    vocab_path = os.path.join(args.save_dir, "vocab-label.pkl")
    pickle.dump(label_vocab, open(vocab_path, "wb"))

    logging.info("Initializing model...")
    crf = M.CRF(len(label_vocab))
    print('args.word_dim==',args.word_dim,type(args.word_dim))
    model = M.LSTMCRF(
        crf=crf,
        vocab_sizes=[len(v) for v in input_vocabs],
        word_dims=args.word_dim,
        hidden_dim=args.lstm_dim,
        layers=args.lstm_layers,
        dropout_prob=args.dropout_prob,
        bidirectional=args.bidirectional
    )
    model.reset_parameters()
    if args.gpu:
        gpu_main = args.gpu[0]
        model = model.cuda(gpu_main)
    params = sum(np.prod(p.size()) for p in model.parameters())
    logging.info("Number of parameters: {}".format(params))

    logging.info("Loading word embeddings...")
    # for vocab, we_type, we_path, we_freeze, emb in \
    #         zip(input_vocabs, args.wordembed_type, args.wordembed_path,
    #             args.wordembed_freeze, model.embeddings):
    #     if we_type == "glove":
    #         assert we_path is not None
    #         load_glove_embeddings(emb, vocab, we_path)
    #     elif we_type == "fasttext":
    #         assert we_path is not None
    #         assert args.fasttext_path is not None
    #         load_fasttext_embeddings(emb, vocab,
    #                                  fasttext_path=args.fasttext_path,
    #                                  embedding_path=we_path)
    #     elif we_type == "none":
    #         pass
    #     else:
    #         raise ValueError("Unrecognized word embedding "
    #                          "type: {}".format(we_type))
    #
    #     if we_freeze:
    #         emb.weight.requires_grad = False

    # Copying configuration file to save directory if config file is specified.
    if args.config:
        config_path = os.path.join(args.save_dir, os.path.basename(args.config))
        shutil.copy(args.config, config_path)

    def create_dataloader(dataset):
        return D.MultiSentWordDataLoader(
            dataset=dataset,
            input_vocabs=input_vocabs,
            label_vocabs=label_vocab,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            tensor_lens=True,
            num_workers=len(args.gpu) if args.gpu is not None else 1,
            pin_memory=True
        )

    dataset = D.MultiSentWordDataset(*args.input_path, args.label_path)
    test_dataset = D.MultiSentWordDataset(*args.test_input_path, args.test_label_path)

    if args.val:
        vr = args.val_ratio
        val_dataset, _ = dataset.split(vr, 1-vr, shuffle=args.shuffle)
    else:
        val_dataset = None

    train_dataset = dataset
    train_dataloader = create_dataloader(train_dataset)
    test_dataloader = create_dataloader(test_dataset)

    if val_dataset is not None:
        val_dataloader = create_dataloader(val_dataset)
    else:
        val_dataloader = None
    print(input_vocabs,type(input_vocabs))

    logging.info("Beginning training...")
    trainer = LSTMCRFTrainer(
        sargs=args,
        input_vocabs=input_vocabs,
        label_vocab=label_vocab,
        val_data=val_dataloader,
        model=model,
        epochs=args.epochs,
        gpus=args.gpu
    )

    trainer.train(train_dataloader, data_size=len(train_dataset))
    # trainer.validate()
    logging.info("Beginning testing...")
    # trainer.test(train_dataloader, data_size=len(train_dataset))
    #trainer.test(test_dataloader, data_size=len(test_dataset))
    logging.info("Done!")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
