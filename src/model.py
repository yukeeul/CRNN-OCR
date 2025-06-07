import nltk
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from .utils import ctc_greedy_decode


class BidirectionalLSTM(nn.Module):
    """Ref: https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/sequence_modeling.py"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size], T = num_steps.
        output : contextual feature [batch_size x T x output_size]
        """
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class CRNN(nn.Module):

    def __init__(self, charset_size, hidden_size):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, 3, 2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0, 2),
            nn.Conv2d(4, 16, 3, 2),
            nn.LeakyReLU(0, 2),
            nn.Conv2d(16, 64, 3, 2),
            nn.LeakyReLU(0, 2),
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(0, 2),
            nn.Conv2d(128, 256, 3),
            nn.LeakyReLU(0, 2),
            nn.Conv2d(256, 512, 3),
            nn.LeakyReLU(0, 2),
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size), BidirectionalLSTM(hidden_size, hidden_size, charset_size)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)

        conv = conv.flatten(2, 3)
        conv = conv.swapaxes(1, 2)

        # rnn features
        output = self.rnn(conv)

        return output


class PLCRNN(pl.LightningModule):
    def __init__(self, charset_size, hidden_size, tokenizer):
        super().__init__()
        self.model = CRNN(
            charset_size=charset_size,
            hidden_size=hidden_size,
        )
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]
        gts = torch.zeros(batch_size, self.tokenizer.max_label_length)
        gt_lens = []
        for i in range(batch_size):
            gt = self.tokenizer.tokenize(labels[i])
            gts[i, : len(gt)] = gt
            gt_lens.append(min(len(labels[i]), self.tokenizer.max_label_length))

        logits = self.model.forward(x)
        log_probs = F.log_softmax(logits, dim=-1)

        loss = F.ctc_loss(
            log_probs.swapaxes(0, 1),
            gts,
            input_lengths=torch.ones(batch_size, dtype=torch.long) * log_probs.shape[1],
            target_lengths=torch.LongTensor(gt_lens),
        )
        self.log("train_loss", loss.item(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]
        gts = torch.zeros(batch_size, self.tokenizer.max_label_length)
        gt_lens = []
        for i in range(batch_size):
            gt = self.tokenizer.tokenize(labels[i])
            gts[i, : len(gt)] = gt
            gt_lens.append(len(labels[i]))

        logits = self.model.forward(x)
        log_probs = F.log_softmax(logits, dim=-1)

        loss = F.ctc_loss(
            log_probs.swapaxes(0, 1),
            gts,
            input_lengths=torch.ones(batch_size, dtype=torch.long) * log_probs.shape[1],
            target_lengths=torch.LongTensor(gt_lens),
        )
        self.log("val_loss", loss.item(), on_epoch=True)

        preds = ctc_greedy_decode(log_probs.detach().argmax(dim=-1).cpu())

        val_ned = 0
        val_accuracy = 0
        for i in range(batch_size):
            gt = labels[i].strip()
            pred = self.tokenizer.decode(preds[i]).strip()

            val_ned += 1 - (nltk.edit_distance(gt, pred) / max(len(gt), len(pred)))
            val_accuracy += gt == pred

        self.log("val_ned", val_ned / batch_size, on_epoch=True)
        self.log("val_accuracy", val_accuracy / batch_size, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        return self.model(x)
