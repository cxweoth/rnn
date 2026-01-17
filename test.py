import re
import os
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset

from test.load_test_dataset import load_test_dataset


# -------------------------
# Config
# -------------------------
SEED = 42
VOCAB_SIZE = 20000        # top-K tokens
MAX_LEN = 200             # truncate/pad length
BATCH_SIZE = 64
EMB_DIM = 128
HIDDEN_SIZE = 128
LR = 1e-3
EPOCHS = 3

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Tokenizer (simple, fast)
# -------------------------
_token_re = re.compile(r"[a-z0-9']+")

def tokenize(text: str):
    # lower + keep a-z0-9'
    return _token_re.findall(text.lower())


# -------------------------
# Vocab
# -------------------------
def build_vocab(texts, vocab_size: int):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    most_common = counter.most_common(vocab_size - 2)  # reserve for pad/unk

    itos = [PAD_TOKEN, UNK_TOKEN] + [w for w, _ in most_common]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(text: str, stoi, max_len: int):
    toks = tokenize(text)
    ids = [stoi.get(w, UNK_IDX) for w in toks[:max_len]]
    # pad
    if len(ids) < max_len:
        ids += [PAD_IDX] * (max_len - len(ids))
    return ids


# -------------------------
# Manual Vanilla RNN using nn.Linear (NO nn.RNN / nn.RNNCell)
# -------------------------
class VanillaRNN(nn.Module):
    """
    h_t = tanh(W_x x_t + W_h h_{t-1} + b)
    logits = W_y h_T + b_y
    """
    def __init__(self, input_dim, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_dim, hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h2y = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x, h0=None):
        """
        x: (B, T, input_dim)
        return: logits (B, num_classes)
        """
        B, T, _ = x.shape
        if h0 is None:
            h = x.new_zeros((B, self.hidden_size))
        else:
            h = h0

        for t in range(T):
            xt = x[:, t, :]
            h = torch.tanh(self.x2h(xt) + self.h2h(h))

        logits = self.h2y(h)
        return logits


class TextRNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = VanillaRNN(emb_dim, hidden_size, num_classes)

    def forward(self, token_ids):
        """
        token_ids: (B, T) int64
        """
        emb = self.embedding(token_ids)  # (B, T, emb_dim)
        logits = self.rnn(emb)           # (B, 2)
        return logits


# -------------------------
# Data Collator
# -------------------------
def make_collate(stoi, max_len):
    def collate(batch):
        # batch: list of dicts with keys: 'text', 'label'
        xs = [encode(ex["text"], stoi, max_len) for ex in batch]
        ys = [ex["label"] for ex in batch]
        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        return x, y
    return collate


# -------------------------
# Train / Eval
# -------------------------
def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, y)

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item()
        total_acc += accuracy(logits, y)

    return total_loss / len(loader), total_acc / len(loader)


def main():
    set_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    # Load dataset (IMDb)
    # get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.join(current_dir, "data", "imdb")
    ds = load_test_dataset(path=load_path)
    train_texts = ds["train"]["text"]

    # Build vocab from train only
    stoi, itos = build_vocab(train_texts, VOCAB_SIZE)
    print("vocab_size:", len(itos))

    print("stoi:", stoi)
    print("itos:", itos)

    # collate_fn = make_collate(stoi, MAX_LEN)
    # train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    # test_loader = DataLoader(ds["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # model = TextRNNClassifier(vocab_size=len(itos), emb_dim=EMB_DIM, hidden_size=HIDDEN_SIZE).to(device)
    # opt = torch.optim.Adam(model.parameters(), lr=LR)

    # for epoch in range(1, EPOCHS + 1):
    #     tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
    #     te_loss, te_acc = eval_one_epoch(model, test_loader, device)
    #     print(f"Epoch {epoch}: train loss={tr_loss:.4f} acc={tr_acc:.4f} | test loss={te_loss:.4f} acc={te_acc:.4f}")


if __name__ == "__main__":
    main()