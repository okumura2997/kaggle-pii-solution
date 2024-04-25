import argparse
import importlib
import json
import sys
from glob import glob

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)

sys.path.append('/kaggle/src')
sys.path.append('/kaggle/configs')

from postprocess import create_pred_df
from preprocess import preprocess_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', type=str)
    parser.add_argument('--max_token_length', type=int, default=None)
    parser.add_argument('--out_file', type=str, default='oof.npy')
    args = parser.parse_args()

    config = importlib.import_module(f'{args.config_name}').config
    # use latest checkpoints
    checkpoint_paths = [sorted(glob(f'/kaggle/outputs/{args.config_name}/fold{i}/checkpoint-*'))[-1] for i in config.folds]
    print(checkpoint_paths)

    model_config = AutoConfig.from_pretrained(checkpoint_paths[0])
    label2id, id2label = model_config.label2id, model_config.id2label
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_paths[0])

    with open('/kaggle/input/pii-detection-removal-from-educational-data/train.json', 'r') as f:
        train = json.load(f)

    if args.max_token_length is not None:
        config.max_token_length = args.max_token_length

    dataset = preprocess_pipeline(train, tokenizer=tokenizer, config=config, label2id=label2id)
    df = dataset.to_pandas()
    df = df[['document', 'spacy_tokens', 'spacy_labels', 'spacy2lm', 'length', 'input_ids', 'labels']].explode(['input_ids', 'labels'])
    dataset = Dataset.from_dict({'document': df.document, 'input_ids': df.input_ids, 'labels': df.labels})

    pred_dfs = []
    for i in range(config.n_folds):
        valid_dataset = dataset.filter(lambda x: x['document'] % config.n_folds == i)
        valid_dataset = valid_dataset.remove_columns('document')
        valid_df = df[df['document'] % config.n_folds == i].copy()

        checkpoint_path = checkpoint_paths[i]
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

        infer_args = TrainingArguments('.', per_device_eval_batch_size=2, report_to='none')
        trainer = Trainer(model=model, args=infer_args, data_collator=collator, tokenizer=tokenizer)

        # inference
        outputs = trainer.predict(valid_dataset)
        pred_df = create_pred_df(outputs, valid_df, config=config)
        pred_dfs.append(pred_df)

    oof_df = pd.concat(pred_dfs, axis=0).sort_values(by=['document', 'token_idx'])
    preds = np.array([pred if pred is not None else [0] * 14 for pred in oof_df['spacy_token_preds']])
    assert preds.shape[0] == 4992533
    np.save(f'/kaggle/outputs/{args.config_name}/{args.out_file}', preds)
