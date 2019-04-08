import sys
import os
import torch
import pickle

import numpy as np

from os.path import join

from torch.nn.functional import softmax
from tqdm import tqdm
from utils import make_path
from model_pytorch import DoubleHeadModel


def iter_data(*datas, batch_size=128, truncate=False, verbose=False, max_batches=float("inf")):
    n_samples = len(datas[0])
    if truncate:
        n_samples = (n_samples // batch_size) * batch_size
    n_samples = min(n_samples, max_batches * batch_size)

    n_batches = 0
    if verbose:
        f = sys.stderr
    else:
        f = open(os.devnull, 'w')
    for i in tqdm(range(0, n_samples, batch_size), total=n_samples // batch_size, file=f, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i: i + batch_size]
        else:
            yield (d[i: i + batch_size] for d in datas)
        n_batches += 1


def iter_apply(X, M, Y, model, loss_fct, device, batch_size):
    logits = []
    cost = 0
    with torch.no_grad():
        model.eval()
        for x, m, y in iter_data(X, M, Y, batch_size=batch_size, truncate=False, verbose=True):
            n = len(x)
            x = torch.tensor(x, dtype=torch.long).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            m = torch.tensor(m).to(device)
            _, clf_logits = model(x)
            #clf_logits *= n
            clf_losses = loss_fct(x, y, m, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost


def iter_predict(X, model, device, batch_size, compute_probs=False):
    logits = []
    probs = []
    with torch.no_grad():
        model.eval()
        for x in iter_data(X, batch_size=batch_size, truncate=False, verbose=True):
            x = torch.tensor(x, dtype=torch.long).to(device)
            _, clf_logits = model(x)
            if compute_probs:
                probs.append(softmax(clf_logits, dim=1).to("cpu").numpy())
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)

    if compute_probs:
        probs = np.concatenate(probs, 0)
        return logits, probs
    else:
        return logits, None


def predict(X, model, device, batch_size, compute_probs=False):
    pred_fn = lambda x: np.argmax(x, 1)
    logits, probs = iter_predict(X, model, device, batch_size, compute_probs=compute_probs)
    predictions = pred_fn(logits)

    return predictions, probs


def persist_model(save_dir, model, text_encoder, label_encoder, model_name='model.pt'):
    model.module.save_to_file(make_path(join(save_dir, model_name)))
    with open(join(save_dir, 'text_encoder.pkl'), 'wb') as f:
        pickle.dump(text_encoder, f)
    with open(join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)


def load_model(save_dir, model_file='model.pt', text_encoder_file='text_encoder.pkl',
               label_encoder_file='label_encoder.pkl'):

    model = DoubleHeadModel.load_from_file(join(save_dir, model_file))
    with open(join(save_dir, text_encoder_file), 'rb') as f:
        text_encoder = pickle.load(f)
    with open(join(save_dir, label_encoder_file), 'rb') as f:
        label_encoder = pickle.load(f)

    return model, text_encoder, label_encoder
