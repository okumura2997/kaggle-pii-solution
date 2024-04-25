import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch

import wandb

sys.path.append('/kaggle/src')
from metrics import compute_metrics_lb
from postprocess import postprocess_preds

ALL_LABELS = [
    'B-EMAIL',
    'B-ID_NUM',
    'B-NAME_STUDENT',
    'B-PHONE_NUM',
    'B-STREET_ADDRESS',
    'B-URL_PERSONAL',
    'B-USERNAME',
    'I-ID_NUM',
    'I-NAME_STUDENT',
    'I-PHONE_NUM',
    'I-STREET_ADDRESS',
    'I-URL_PERSONAL',
    'I-USERNAME',
    'O'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_ids', nargs='+', type=str)
    parser.add_argument('--o_thr_init', default=0.87)

    args = parser.parse_args()
    print(args)

    wandb.init(
        project='pii-detection-removal-from-educational-data',
        job_type='ensemble_per_class',
        config=args
    )

    label2id = {label: idx for idx, label in enumerate(ALL_LABELS)}
    id2label = {v: k for k, v in label2id.items()}

    with open('/kaggle/input/pii-detection-removal-from-educational-data/train.json', 'r') as f:
        train = json.load(f)

    train_df = pd.DataFrame(train)[['document', 'tokens', 'labels']]
    train_df = train_df.explode(['tokens', 'labels'])
    train_df['token'] = train_df.groupby('document').cumcount()
    train_df = train_df.sort_values(by=['document', 'token'])

    gt_df = train_df.copy()
    gt_df = gt_df[gt_df['labels'] != 'O']
    gt_df['row_id'] = np.arange(len(gt_df))
    gt_df = gt_df.rename(columns={'labels': 'label'})
    gt_df = gt_df[['row_id', 'document', 'token', 'tokens', 'label']]

    oofs = np.stack([np.load(f'/kaggle/outputs/{exp_id}/oof.npy') for exp_id in args.exp_ids])
    oofs[(oofs.sum(axis=2) == 0)] = np.nan


    def objective(x, oofs=oofs, train_df=train_df, gt_df=gt_df):
        o_thr = x[0]

        thr_per_type_dict = {
            'B-EMAIL': x[1],
            'B-ID_NUM': x[2],
            'I-ID_NUM': x[3],
            'B-NAME_STUDENT': x[4],
            'I-NAME_STUDENT': x[5],
            'B-PHONE_NUM': x[6],
            'I-PHONE_NUM': x[7],
            'B-URL_PERSONAL': x[8],
            'I-URL_PERSONAL': x[9],
            'B-USERNAME': x[10],
            'I-USERNAME': x[11],
            'B-STREET_ADDRESS': x[12],
            'I-STREET_ADDRESS': x[13]
        }

        preds = np.nanmean(oofs, axis=0)
        is_nans = np.isnan(preds).sum(axis=1) == len(ALL_LABELS)
        o_probas = preds[:, label2id['O']]
        preds_without_o = preds[:, :label2id['O']]
        preds_without_o = torch.from_numpy(preds_without_o * 1e4).softmax(axis=1).numpy()

        o_id = label2id['O']
        pred_labels = []
        for i in range(len(preds)):
            pred = preds[i]
            is_nan = is_nans[i]
            o_proba = o_probas[i]
            pred_without_o = preds_without_o[i]

            if is_nan:
                pred_label = o_id
            else:
                if o_proba >= o_thr:
                    pred_label = o_id
                else:
                    _pred_label = pred_without_o.argmax()
                    if id2label[_pred_label] in thr_per_type_dict.keys():
                        thr = thr_per_type_dict[id2label[_pred_label]]
                        pred_label_proba = pred[_pred_label]
                        pred_label = _pred_label if pred_label_proba > thr else o_id
                    else:
                        pred_label = _pred_label

            pred_labels.append(pred_label)

        df_ens = train_df.copy()
        df_ens['pred_label'] = pred_labels
        df_ens['pred_label'] = df_ens['pred_label'].map(id2label)
        df_ens = df_ens[df_ens['pred_label'] != 'O']
        df_ens = df_ens[['document', 'token', 'pred_label']].rename(columns={'pred_label': 'label'})
        df_ens['row_id'] = np.arange(len(df_ens))
        df_ens = df_ens[['row_id', 'document', 'token', 'label']].reset_index(drop=True)

        metrics_lb = compute_metrics_lb(df_ens, gt_df)

        return 1 - metrics_lb['ents_f5']

    x_init = [args.o_thr_init] + [0.5, 0.4, 0.4, 0.07, 0.07, 0.5, 0.5, 0.25, 0.25, 0.4, 0.4, 0.4, 0.4]

    result = minimize(fun=objective, x0=x_init, method='Nelder-Mead')
    print(result.fun)
    wandb.log({'result': result})
    wandb.finish()
