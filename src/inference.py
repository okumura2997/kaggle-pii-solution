import importlib
import json
import sys

import numpy as np
from datasets import Dataset
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)

sys.path.append('/kaggle/src')
sys.path.append('/kaggle/configs')

from postprocess import create_pred_df, postprocess_preds
from preprocess import preprocess_pipeline


exp_id = 'exp005'
checkpoint_path = '/kaggle/outputs/exp005/fold0/checkpoint-3200'
json_path = '/kaggle/input/pii-detection-removal-from-educational-data/test.json'
o_threshold = 0.9


def inference_pipeline(json_path, config, checkpoint_path, label2id):
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    args = TrainingArguments(
        ".",
        per_device_eval_batch_size=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        tokenizer=tokenizer
    )

    with open(json_path) as f:
        test = json.load(f)

    # preproces
    test_dataset = preprocess_pipeline(test, tokenizer, config, label2id, is_train=False)
    test_df = test_dataset.to_pandas()[['document', 'spacy_tokens', 'spacy_labels', 'spacy2lm', 'length', 'input_ids']]
    test_df = test_df.explode('input_ids')
    test_dataset = Dataset.from_dict({'input_ids': test_df.input_ids})

    # inference
    outputs = trainer.predict(test_dataset)
    pred_df = create_pred_df(outputs, test_df, config=config)

    return pred_df


if __name__ == '__main__':
    config = importlib.import_module(exp_id).config
    model_config = AutoConfig.from_pretrained(checkpoint_path)

    id2label = model_config.id2label
    label2id = model_config.label2id

    pred_df = inference_pipeline(json_path, config, checkpoint_path, label2id)

    # postprocess
    pred_df['pred_label'] = pred_df['spacy_token_preds'].map(
        lambda x: id2label[postprocess_preds(x, label2id=label2id, o_threshold=o_threshold)]
    )

    # fix up dataframe
    pred_df = pred_df[pred_df['pred_label'] != 'O']
    pred_df = pred_df[['document', 'token_idx', 'pred_label']].rename(columns={'token_idx': 'token', 'pred_label': 'label'})
    pred_df['row_id'] = np.arange(len(pred_df))
    pred_df = pred_df[['row_id', 'document', 'token', 'label']].reset_index(drop=True)
    print(pred_df)
