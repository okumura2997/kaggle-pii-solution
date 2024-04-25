import numpy as np
import pandas as pd
import torch
from transformers.trainer_utils import PredictionOutput


def merge_sliding_window_predictions(predictions, valid_df, max_length, stride):
    pred_df = valid_df.copy()
    pred_df['preds'] = [pred for pred in predictions]

    pred_dfs = []
    for document, df_i in pred_df.groupby('document'):
        spacy_tokens = df_i['spacy_tokens'].to_list()[0]
        spacy_labels = df_i['spacy_labels'].to_list()[0]
        spacy2lm = df_i['spacy2lm'].to_list()[0]

        input_ids_list = df_i['input_ids'].to_list()
        preds_list = df_i['preds'].to_list()

        n_windows = len(input_ids_list)
        n_token_length = df_i['length'].values[0][0]  # token length without special tokens

        input_ids_all_token = [[] for _ in range(n_token_length)]
        preds_all_token = [[] for _ in range(n_token_length)]

        for i in range(n_windows):
            input_ids_i = input_ids_list[i][1:-1]  # remove special tokens
            preds_i = preds_list[i][1:-1]

            offset = i * (max_length - 2 - stride)
            for j in range(len(input_ids_i)):
                input_ids_all_token[offset + j].append(input_ids_i[j])
                preds_all_token[offset + j].append(preds_i[j])

        assert all([len(set(input_ids)) == 1 for input_ids in input_ids_all_token])

        preds_all_token = np.array([np.mean(preds, axis=0) for preds in preds_all_token])

        pred_dfs.append(pd.DataFrame(
            {'document': document, 'spacy_tokens': [spacy_tokens], 'spacy_labels': [spacy_labels],
             'spacy2lm': [spacy2lm], 'preds': [preds_all_token]}
        ))

    pred_df = pd.concat(pred_dfs, axis=0).reset_index(drop=True)

    return pred_df


def get_spacy_token_prediction(pred_df):
    out_df = pred_df.copy()

    spacy_token_preds_list = []
    for _, row in out_df.iterrows():
        spacy_token_preds = []
        spacy2lm = row['spacy2lm']

        preds = row['preds']
        for spacy_token_idx, lm_token_indices in enumerate(spacy2lm):
            if len(lm_token_indices) == 0:
                spacy_token_preds.append(None)
            else:
                spacy_token_preds.append(preds[lm_token_indices[0]])
        spacy_token_preds_list.append(spacy_token_preds)

    out_df['spacy_token_preds'] = spacy_token_preds_list

    return out_df[['document', 'spacy_tokens', 'spacy_labels', 'spacy_token_preds']]


def postprocess_preds(pred, label2id, id2label, o_thr, thr_per_type_dict={}):
    o_id = label2id['O']
    is_nan = all(np.isnan(pred))
    o_proba = pred[o_id]
    pred_without_o = pred[:o_id]
    pred_without_o = torch.from_numpy(pred_without_o * 1e4).softmax(dim=0).numpy()

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

    return pred_label


def create_pred_df(outputs: PredictionOutput, valid_df: pd.DataFrame, config):
    predictions = outputs.predictions
    predictions = torch.from_numpy(predictions).softmax(axis=2).numpy()

    pred_df = merge_sliding_window_predictions(predictions, valid_df, config.max_token_length, config.stride)
    pred_df = get_spacy_token_prediction(pred_df)

    pred_df = pred_df.explode(['spacy_tokens', 'spacy_labels', 'spacy_token_preds'])
    pred_df['token_idx'] = pred_df.groupby('document').cumcount()
    pred_df = pred_df[['document', 'token_idx', 'spacy_tokens', 'spacy_labels', 'spacy_token_preds']]
    # pred_df = pred_df[~pred_df['spacy_token_preds'].isnull()]

    return pred_df


def create_gt_df(df):
    df = df.drop_duplicates(subset=['document'])
    gt_df = df[['document', 'spacy_tokens', 'spacy_labels']].copy()
    gt_df = gt_df.rename(columns={'spacy_tokens': 'token', 'spacy_labels': 'label'})
    gt_df = gt_df.explode(['token', 'label'])
    gt_df['token'] = gt_df.groupby('document').cumcount()

    gt_df = gt_df[gt_df['label'] != 'O'].reset_index(drop=True)
    gt_df['row_id'] = np.arange(len(gt_df))
    gt_df = gt_df[['row_id', 'document', 'token', 'label']].copy()

    return gt_df