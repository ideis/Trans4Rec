import numpy as np
import scipy.stats as st

def evaluate_scores(scores, targets, k=10):
    preds = scores.detach().cpu().numpy()
    item = targets.detach().cpu().numpy()
    ranks = st.rankdata(preds)
    pred_items = preds.argsort()[:k][0]
    recall = int(item in pred_items)
    mrr = (1.0 / ranks[item]).mean()
    return  recall, mrr, pred_items
