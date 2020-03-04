import numpy as np
from torch import Tensor
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, hamming_loss, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import pdb

CLASSIFICATION_THRESHOLD: float = 0.5  # Best keep it in [0.0, 1.0] range
    
labels_list = ['P1','P2','P3','P4','P5']

encoder=LabelBinarizer()

# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1)
#     return np.sum(outputs == labels)


def recall_macro(y_pred: Tensor, y_true: Tensor):
    y_pred = y_pred.cpu()
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)
    y_true = y_true.cpu()
    y_true = np.argmax(y_true, axis=1)
    print(y_true)
    return recall_score(y_pred, y_true, average='macro')

def recall_by_class(y_pred: Tensor, y_true: Tensor, labels: list = labels_list):
    y_pred = y_pred.cpu()
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)
    y_true = y_true.cpu()
    y_true = np.argmax(y_true, axis=1)
    d = {}
    for i in range(len(labels)): 
        out_pred = []
        out_true = []
        for j in y_pred:
            if j == i:
                out_pred.append(1)
            else:
                out_pred.append(0)
        print(out_pred)
        for j in y_true:
            if j == i:
                out_true.append(1)
            else:
                out_true.append(0)
        d[labels[i]] = recall_score(out_pred, out_true, average='micro')
    return d

def recall_micro(y_pred: Tensor, y_true: Tensor):
    y_pred = y_pred.sigmoid()
    y_pred = (y_pred > 0.5).cpu()
    y_true = y_true.cpu()
    return recall_score(y_pred, y_true, average='micro')

def recall_multilabel(y_pred: Tensor, y_true: Tensor, labels:list = labels_list):
    y_pred = y_pred.sigmoid()
    y_pred = (y_pred > 0.5).cpu()
    y_true = y_true.cpu()
    return recall_score(y_pred, y_true, average=None, labels = labels)

def precision_macro(y_pred: Tensor, y_true: Tensor):
    y_pred = y_pred.sigmoid()
    y_pred = (y_pred > 0.5).cpu()
    y_true = y_true.cpu()
    return precision_score(y_pred, y_true, average='macro')

def precision_micro(y_pred: Tensor, y_true: Tensor):
    y_pred = y_pred.sigmoid()
    y_pred = (y_pred > 0.5).cpu()
    y_true = y_true.cpu()
    return precision_score(y_pred, y_true, average='micro')

def precision_multilabel(y_pred: Tensor, y_true: Tensor, labels: list = labels_list):
    y_pred = y_pred.sigmoid()
    y_pred = (y_pred > 0.5).cpu()
    y_true = y_true.cpu()
    return precision_score(y_pred, y_true, average=None, labels = labels)

def f1_macro(y_pred: Tensor, y_true: Tensor):
    y_pred = y_pred.sigmoid()
    y_pred = (y_pred > 0.5).cpu()
    y_true = y_true.cpu()
    return f1_score(y_pred, y_true, average='macro')

def f1_micro(y_pred: Tensor, y_true: Tensor):
    y_pred = y_pred.sigmoid()
    y_pred = (y_pred > 0.5).cpu()
    y_true = y_true.cpu()
    return f1_score(y_pred, y_true, average='micro')

def f1_multilabel(y_pred: Tensor, y_true: Tensor, labels: list = labels_list):
    y_pred = y_pred.sigmoid()
    y_pred = (y_pred > 0.5).cpu()
    y_true = y_true.cpu()
    F1_by_class_d = {}
    for i in range(len(labels)):
        F1_by_class_d[labels[i]] = f1_score(y_true, y_pred, average = 'micro', labels = [i]) # pos_label = i,
    return F1_by_class_d


def accuracy(y_pred: Tensor, y_true: Tensor):
    y_pred = y_pred.cpu()
    outputs = np.argmax(y_pred, axis=1)
    return np.mean(outputs.numpy() == y_true.detach().cpu().numpy())


def accuracy_multilabel(y_pred: Tensor, y_true: Tensor, sigmoid: bool = True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    outputs = np.argmax(y_pred, axis=1)
    real_vals = np.argmax(y_true, axis=1)
    return np.mean(outputs.numpy() == real_vals.numpy())


def accuracy_thresh(
    y_pred: Tensor,
    y_true: Tensor,
    thresh: float = CLASSIFICATION_THRESHOLD,
    sigmoid: bool = True,
):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh) == y_true.bool()).float().mean().item()


#     return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(
    y_pred: Tensor,
    y_true: Tensor,
    thresh: float = 0.3,
    beta: float = 2,
    eps: float = 1e-9,
    sigmoid: bool = True,
):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    rec = TP / (y_true.sum(dim=1) + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean().item()


def roc_auc(y_pred: Tensor, y_true: Tensor):
    # ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc["micro"]


def Hamming_loss(
    y_pred: Tensor,
    y_true: Tensor,
    sigmoid: bool = True,
    thresh: float = CLASSIFICATION_THRESHOLD,
    sample_weight=None,
):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).cpu()
    y_true = y_true.cpu()
    return hamming_loss(y_true, y_pred, sample_weight=sample_weight)


def Exact_Match_Ratio(
    y_pred: Tensor,
    y_true: Tensor,
    sigmoid: bool = True,
    thresh: float = CLASSIFICATION_THRESHOLD,
    normalize: bool = True,
    sample_weight=None,
):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    return accuracy_score(
        y_true, y_pred, normalize=normalize, sample_weight=sample_weight
    )


def F1(y_pred: Tensor, y_true: Tensor, threshold: float = CLASSIFICATION_THRESHOLD):
    return fbeta(y_pred, y_true, thresh=threshold, beta=1)

def roc_auc_score_by_class(y_pred:Tensor, y_true:Tensor, labels:list = labels_list):
    y_pred = y_pred.cpu()
    y_pred = np.argmax(y_pred, axis = 1).numpy()
    y_true = y_true.cpu()
    y_true = y_true.detach().cpu().numpy()
    roc_auc_score_d = {}
    for i in range(len(labels)):
        lb = LabelBinarizer()
        y_true_i = y_true.copy()
        y_true_i[y_true != i] = len(labels) + 1
        y_true_i = lb.fit_transform(y_true_i)
        y_pred_i = y_pred.copy()
        y_pred_i[y_pred != i] = len(labels) + 1
        y_pred_i = lb.transform(y_pred_i)
        roc_auc_score_d[labels[i]] = roc_auc_score(y_true_i, y_pred_i, average = 'micro')
    return roc_auc_score_d
