import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
    parser.add_argument('--optimize_o_thr_only', action='store_true')
    parser.add_argument('--optimize_weights_only', action='store_true')
    parser.add_argument('--o_thr', default=-1, type=float)

    args = parser.parse_args()
    print(args)

    wandb.init(
        project='pii-detection-removal-from-educational-data',
        job_type='ensemble',
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

    if args.optimize_o_thr_only:
        x_init = [args.o_thr_init]

        def objective(x, oofs=oofs, train_df=train_df, gt_df=gt_df):
            weights = [1] * len(oofs)
            weights = np.array(weights)

            oof_ens = (weights[:, np.newaxis, np.newaxis] * oofs)
            oof_ens = np.nanmean(oof_ens, axis=0)

            o_thr = x[0]

            pred_labels = []
            for pred in oof_ens:
                if all(np.isnan(pred)):
                    pred_label = label2id['O']
                else:
                    pred_label = postprocess_preds(pred, label2id=label2id, o_threshold=o_thr)
                pred_labels.append(pred_label)

            # fix up dataframe
            df_ens = train_df.copy()
            df_ens['pred_label'] = pred_labels
            df_ens['pred_label'] = df_ens['pred_label'].map(id2label)
            df_ens = df_ens[df_ens['pred_label'] != 'O']
            df_ens = df_ens[['document', 'token', 'pred_label']].rename(columns={'pred_label': 'label'})
            df_ens['row_id'] = np.arange(len(df_ens))
            df_ens = df_ens[['row_id', 'document', 'token', 'label']].reset_index(drop=True)

            metrics_lb = compute_metrics_lb(df_ens, gt_df)

            return 1 - metrics_lb['ents_f5']

    elif args.optimize_weights_only:
        x_init = [1] * len(args.exp_ids)
        assert args.o_thr != -1, 'o_thr is needed when optimizing weights only'

        def objective(x, oofs=oofs, train_df=train_df, gt_df=gt_df, o_thr=args.o_thr):
            weights = x
            weights = np.array(weights)

            oof_ens = (weights[:, np.newaxis, np.newaxis] * oofs)
            oof_ens = np.nanmean(oof_ens, axis=0)

            pred_labels = []
            for pred in oof_ens:
                if all(np.isnan(pred)):
                    pred_label = label2id['O']
                else:
                    pred_label = postprocess_preds(pred, label2id=label2id, o_threshold=o_thr)
                pred_labels.append(pred_label)

            # fix up dataframe
            df_ens = train_df.copy()
            df_ens['pred_label'] = pred_labels
            df_ens['pred_label'] = df_ens['pred_label'].map(id2label)
            df_ens = df_ens[df_ens['pred_label'] != 'O']
            df_ens = df_ens[['document', 'token', 'pred_label']].rename(columns={'pred_label': 'label'})
            df_ens['row_id'] = np.arange(len(df_ens))
            df_ens = df_ens[['row_id', 'document', 'token', 'label']].reset_index(drop=True)

            metrics_lb = compute_metrics_lb(df_ens, gt_df)

            return 1 - metrics_lb['ents_f5']

    else:
        x_init = [1] * len(args.exp_ids)
        x_init.append(args.o_thr_init)

        def objective(x, oofs=oofs, train_df=train_df, gt_df=gt_df):
            weights = x[:-1]
            weights = np.array(weights)

            oof_ens = (weights[:, np.newaxis, np.newaxis] * oofs)
            oof_ens = np.nanmean(oof_ens, axis=0)

            o_thr = x[-1]

            pred_labels = []
            for pred in oof_ens:
                if all(np.isnan(pred)):
                    pred_label = label2id['O']
                else:
                    pred_label = postprocess_preds(pred, label2id=label2id, o_threshold=o_thr)
                pred_labels.append(pred_label)

            # fix up dataframe
            df_ens = train_df.copy()
            df_ens['pred_label'] = pred_labels
            df_ens['pred_label'] = df_ens['pred_label'].map(id2label)
            df_ens = df_ens[df_ens['pred_label'] != 'O']
            df_ens = df_ens[['document', 'token', 'pred_label']].rename(columns={'pred_label': 'label'})
            df_ens['row_id'] = np.arange(len(df_ens))
            df_ens = df_ens[['row_id', 'document', 'token', 'label']].reset_index(drop=True)

            metrics_lb = compute_metrics_lb(df_ens, gt_df)

            return 1 - metrics_lb['ents_f5']

    result = minimize(fun=objective, x0=x_init, method='Nelder-Mead')
    wandb.log({'result': result})
    wandb.finish()
