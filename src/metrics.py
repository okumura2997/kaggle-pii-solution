from collections import defaultdict
from typing import Dict

import evaluate
import numpy as np
from postprocess import create_gt_df, create_pred_df, postprocess_preds
from sklearn.metrics import classification_report


def compute_metrics_per_token(predictions, label_ids, label2id):
    mask = label_ids != -100
    result = classification_report(
        label_ids[mask], predictions[mask], zero_division=0,
        labels=list(label2id.values()), target_names=list(label2id.keys()),
        output_dict=True
    )

    if 'micro avg' in result.keys():
        result.pop('micro avg')

    result.pop('macro avg')
    result.pop('weighted avg')

    # add suffix
    result = {key + '_token': value for key, value in result.items()}

    return result


def compute_metrics_per_entity(predictions, label_ids, id2label):
    seqeval = evaluate.load('seqeval')

    predictions_str, labels_str = [], []
    for prediction, label in zip(predictions, label_ids):
        predictions_str_i, labels_str_i = [], []
        for p, l in zip(prediction, label):
            if l != -100:
                predictions_str_i.append(id2label[p])
                labels_str_i.append(id2label[l])
        predictions_str.append(predictions_str_i)
        labels_str.append(labels_str_i)

    result = seqeval.compute(predictions=predictions_str, references=labels_str, zero_division=0)

    precision = result['overall_precision']
    recall = result['overall_recall']
    f5 = (1 + 5*5) * recall * precision / (5*5*precision + recall + 1e-100)
    result['overall_f5'] = f5

    # add suffix
    result = {key + '_entity': value for key, value in result.items()}

    return result


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


def compute_metrics_lb(pred_df, gt_df):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """

    references = {(row.document, row.token, row.label) for row in gt_df.itertuples()}
    predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

    score_per_type = defaultdict(PRFScore)
    references = set(references)

    for ex in predictions:
        pred_type = ex[-1] # (document, token, label)
        if pred_type != 'O':
            pred_type = pred_type[2:] # avoid B- and I- prefix

        if pred_type not in score_per_type:
            score_per_type[pred_type] = PRFScore()

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1

    for doc, tok, ref_type in references:
        if ref_type != 'O':
            ref_type = ref_type[2:] # avoid B- and I- prefix

        if ref_type not in score_per_type:
            score_per_type[ref_type] = PRFScore()
        score_per_type[ref_type].fn += 1

    totals = PRFScore()

    for prf in score_per_type.values():
        totals += prf

    return {
        "ents_p": totals.precision,
        "ents_r": totals.recall,
        "ents_f5": totals.f5,
        "ents_per_type": {k: v.to_dict() for k, v in score_per_type.items() if k!= 'O'},
    }


def compute_metrics(outputs, valid_df, config, label2id, id2label):
    # predictions = outputs.predictions
    # predictions = torch.from_numpy(predictions).softmax(axis=2).numpy()
    # predictions = predictions.argmax(axis=2)

    # label_ids = outputs.label_ids

    # result = compute_metrics_per_token(predictions, label_ids, label2id=label2id)
    # result_entity = compute_metrics_per_entity(predictions, label_ids, id2label=id2label)
    # result.update(result_entity)

    result = {}

    gt_df = create_gt_df(valid_df)
    pred_df = create_pred_df(outputs, valid_df, config=config)
    pred_df = pred_df[~pred_df['spacy_token_preds'].isnull()]

    pred_dfs = []
    for o_threshold in config.o_thresholds:
        pred_df_i = pred_df.copy()
        pred_df_i['pred_label'] = pred_df_i['spacy_token_preds'].map(
            lambda x: id2label[postprocess_preds(x, label2id=label2id, o_threshold=o_threshold)]
        )

        # fix up dataframe
        pred_df_i = pred_df_i[pred_df_i['pred_label'] != 'O']
        pred_df_i = pred_df_i[['document', 'token_idx', 'pred_label']].rename(columns={'token_idx': 'token', 'pred_label': 'label'})
        pred_df_i['row_id'] = np.arange(len(pred_df_i))
        pred_df_i = pred_df_i[['row_id', 'document', 'token', 'label']].reset_index(drop=True)
        pred_dfs.append(pred_df_i)

    for o_threshold, pred_df in zip(config.o_thresholds, pred_dfs):
        result_lb = compute_metrics_lb(pred_df, gt_df)
        # add suffix
        result_lb = {key + f'_{o_threshold:.2f}': value for key, value in result_lb.items()}
        result.update(result_lb)

    return result
