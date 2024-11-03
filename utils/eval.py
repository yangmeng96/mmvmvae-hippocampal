import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

import torch

from clfs.rats_clf import ClfRats


def train_clf_lr(encodings, labels):
    clf = LogisticRegression(max_iter=10000).fit(encodings.cpu(), labels.cpu())
    return clf


def eval_clf_lr(clf, encodings, labels):
    y_pred = clf.predict(encodings.cpu())
    # bal_acc = balanced_accuracy_score(labels.cpu(), y_pred, adjusted=True)
    bal_acc = accuracy_score(labels.cpu(), y_pred)
    return np.array(bal_acc)

def train_clf_lr_cub(encodings, labels):
    n_labels = labels.shape[1]
    clfs = []
    for k in range(0, n_labels):
        clf = LogisticRegression(max_iter=10000).fit(
            encodings.cpu(), labels[:, k].cpu()
        )
        clfs.append(clf)
    return clfs

def eval_clf_lr_cub(clfs, encodings, labels):
    n_labels = labels.shape[1]
    scores = torch.zeros(n_labels)
    for k in range(0, n_labels):
        clf_k = clfs[k]
        y_pred_k = clf_k.predict(encodings.cpu())
        ap = average_precision_score(labels[:, k].cpu(), y_pred_k)
        scores[k] = ap
    return scores

def generate_samples(decoders, rep):
    imgs_gen = []
    for dec in decoders:
        img_gen = dec(rep)
        imgs_gen.append(img_gen[0])
    return imgs_gen


def conditional_generation(mvvae, dists, decoders):
    imgs_gen = []
    for idx, dist in enumerate(dists):
        mu, lv = dist
        imgs_gen_dist = []
        for m in range(len(decoders)):
            z_out = mvvae.reparametrize(mu, lv)
            cond_gen_m = decoders[m](z_out)[0]
            imgs_gen_dist.append(cond_gen_m)
        imgs_gen.append(imgs_gen_dist)
    return imgs_gen

def load_modality_clfs_rats(cfg):
    fp_clf = os.path.join(cfg.dataset.dir_clfs_base, cfg.dataset.suffix_clfs, "last.ckpt")
    model = ClfRats.load_from_checkpoint(fp_clf)
    return model

def calc_coherence(clf, imgs, labels):
    out_clf = clf([imgs, labels])
    accs = torch.zeros(1, len(imgs.keys()))
    for m in range(len(imgs.keys())):
        pred_m = out_clf[0][:, m, :]
        acc_m = accuracy_score(labels.cpu(), np.argmax(pred_m.cpu().numpy(), axis=1).astype(int))
        accs[0, m] = acc_m
    return accs

def calc_coherence_ap(cfg, clf, imgs, labels):
    n_labels = labels.shape[1]
    out_clf = clf(cfg, [imgs, labels])
    aps = torch.zeros(1, len(imgs.keys()), n_labels)
    for m in range(len(imgs.keys())):
        for k in range(0, n_labels):
            pred_m_k = out_clf[0][:, m, k]
            ap_m_k = average_precision_score(
                labels[:, k].cpu(), 
                pred_m_k.detach().cpu().numpy()
            )
            aps[0, m, k] = ap_m_k
    return aps