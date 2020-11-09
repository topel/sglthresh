import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import KFold

import pickle
import time
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device, torch.cuda.get_device_name(0))

print("pytorch version:", torch.__version__)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True


def print_scores(gt_y, preds):
    print(classification_report(gt_y, preds, digits=3))
    set_accuracy = accuracy_score(gt_y, preds)
    print('set acc: %.3f' % (set_accuracy))


def print_thresholds(thresh, nb_classes):
    s = ''
    for k in range(nb_classes):
        s += '%.3f ' % (thresh[k])
    print(s)


def save_thresholds_to_text(thresh, fpath):
    nb_classes = len(thresh)
    print(nb_classes)
    with open(fpath, 'wt') as fh:
        for k in range(nb_classes):
            fh.write("%.6f\n" % (thresh[k]))


def save_thresholds(thresh, fpath):
    np.save(fpath, np.array(thresh))


def load_thresholds(fpath):
    return np.load(fpath)


def binarize_probs(probs, thresholds):
    nb_classes = probs.shape[-1]
    binarized_output = np.zeros_like(probs)

    for k in range(nb_classes):
        binarized_output[:, k] = (np.sign(probs[:, k] - thresholds[k]) + 1) // 2

    return binarized_output

# Cnn14
val_pred_np = torch.load('datasets/audioset/Cnn14/eval_clipwise_predictions.pth')
val_gt_np = torch.load('datasets/audioset/Cnn14/eval_target.pth')

nb_classes=val_gt_np.shape[1]
print(nb_classes)

s = np.sum(val_gt_np, axis=1)
keep_ind = np.where(s > 0)
# new_val_gt_np = val_gt_np.copy()
val_gt_np = val_gt_np[keep_ind]
val_pred_np = val_pred_np[keep_ind]
# s = np.sum(new_val_gt_np, axis=1)
# new_val_gt_np.shape, np.where(s < 1)
print(val_gt_np.shape, val_pred_np.shape)

kf = KFold(n_splits=3)
kf.get_n_splits(val_gt_np)

print(kf)

fold_ind = 0
for val_index, test_index in kf.split(val_gt_np):
    val_gt_np_fold, eval_gt_np_fold = val_gt_np[val_index], val_gt_np[test_index]
    val_pred_np_fold, eval_pred_np_fold = val_pred_np[val_index], val_pred_np[test_index]
    print("fold:", fold_ind, "VAL:", val_index.shape, "TEST:", test_index.shape)
    fold_ind += 1


from heu_utils.heuThresh import heu_threshold_opti, heu_threshold_opti_meanF1

mean_micro_f1 = [0, 0]

fold_ind = 0

for val_index, test_index in kf.split(val_gt_np):
    time1 = time.time()

    t = 0.5
    thresh = [t] * nb_classes

    val_gt_np_fold, eval_gt_np_fold = val_gt_np[val_index], val_gt_np[test_index]
    val_pred_np_fold, eval_pred_np_fold = val_pred_np[val_index], val_pred_np[test_index]

    print("fold:", fold_ind, "VAL:", val_index.shape, "TEST:", test_index.shape)

    # sk_f1, best_f1, learned_AT_thresholds, history = heu_threshold_opti(val_gt_np_fold, val_pred_np_fold, thresh)
    sk_f1, best_f1, learned_AT_thresholds, history = heu_threshold_opti_meanF1(val_gt_np_fold, val_pred_np_fold, thresh, average='micro')

    val_bin_pred = binarize_probs(val_pred_np_fold, learned_AT_thresholds)
    test_bin_pred = binarize_probs(eval_pred_np_fold, learned_AT_thresholds)

    #     print_scores(val_gt_np_fold, val_bin_pred)
    f1 = f1_score(val_gt_np_fold, val_bin_pred, average='micro')
    mean_micro_f1[0] += f1
    print(" val: %.3f" % f1)

    #     print_scores(eval_gt_np_fold, test_bin_pred)
    f1 = f1_score(eval_gt_np_fold, test_bin_pred, average='micro')
    print(" test: %.3f" % f1)
    mean_micro_f1[1] += f1

    fold_ind += 1

    save_thresholds(learned_AT_thresholds, "exp/audioset/heu_meanF1_micro_thresholds_fold%d.npy"%(fold_ind))

    optimizing_time = time.time() - time1
    print('optimizing time: {:.3f} s'
          ''.format(optimizing_time))

print("Average micro F1 (val, eval): %.3f %.3f" % (mean_micro_f1[0] / fold_ind, mean_micro_f1[1] / fold_ind))


# optimizing time: 41472.337 s


