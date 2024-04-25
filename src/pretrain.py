import argparse
import importlib
import json
import os
import sys

from datasets import Dataset, concatenate_datasets
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, set_seed)

import wandb

sys.path.append('/kaggle/src')
sys.path.append('/kaggle/configs')

from preprocess import preprocess_pipeline
from custom_trainer import FocalLossTrainer

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

def freeze(model, config):
    if config.freeze_embeddings:
        if 'deberta' in config.model_path:
            for param in model.deberta.embeddings.parameters():
                param.requires_grad = False
        elif 'longformer' in config.model_path:
            for param in model.longformer.embeddings.parameters():
                param.requires_grad = False

    if config.freeze_layers > 0:
        if 'deberta' in config.model_path:
            for layer in model.deberta.encoder.layer[:config.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        elif 'longformer' in config.model_path:
            for layer in model.longformer.encoder.layer[:config.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', type=str)
    args = parser.parse_args()

    config = importlib.import_module(f'{args.config_name}').config
    train_args = importlib.import_module(f'{args.config_name}').train_args

    set_seed(config.seed)

    label2id = {label: idx for idx, label in enumerate(ALL_LABELS)}
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    if hasattr(config, 'pad_to_multiple_of'):
        collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=config.pad_to_multiple_of)
    else:
        collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    ex_dataset_list = []
    for path in config.external_data_paths:
        with open(path, mode='r') as f:
            ex_data = json.load(f)

        ex_dataset = preprocess_pipeline(ex_data, tokenizer=tokenizer, config=config, label2id=label2id)
        ex_df = ex_dataset.to_pandas()
        ex_df = ex_df[['document', 'spacy_tokens', 'spacy_labels', 'spacy2lm', 'length', 'input_ids', 'labels']].explode(['input_ids', 'labels'])
        ex_dataset = Dataset.from_dict({'input_ids': ex_df.input_ids, 'labels': ex_df.labels})

        ex_dataset_list.append(ex_dataset)

    output_dir = os.path.join(f'/kaggle/outputs/{args.config_name}')
    train_args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(
        project='pii-detection-removal-from-educational-data',
        group=f'{args.config_name}',
        name=f'{args.config_name}',
        config=config
    )

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    model = freeze(model, config)

    train_dataset = concatenate_datasets(ex_dataset_list)

    if hasattr(config, 'loss'):
        if config.loss == 'focal':
            trainer = FocalLossTrainer(
                model=model,
                args=train_args,
                train_dataset=train_dataset,
                data_collator=collator,
                tokenizer=tokenizer,
            )
        else:
            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=train_dataset,
                data_collator=collator,
                tokenizer=tokenizer,
            )

    trainer.train()
    wandb.finish()
