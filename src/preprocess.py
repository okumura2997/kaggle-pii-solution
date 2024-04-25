import numpy as np
from datasets import Dataset
import spacy_alignments


def get_alignments(data, tokenizer):
    spacy_tokens = data['spacy_tokens']
    lm_tokens = tokenizer.convert_ids_to_tokens(data['input_ids'])

    spacy2lm, lm2spacy = spacy_alignments.get_alignments(spacy_tokens, lm_tokens)

    return {'spacy2lm': spacy2lm, 'lm2spacy': lm2spacy}


def align_labels(data, label2id):
    spacy_tokens = data['spacy_tokens']
    spacy_labels = data['spacy_labels']

    spacy2lm = data['spacy2lm']

    labels = np.array([-100] * len(data['input_ids']))
    for i in range(len(spacy_tokens)):
        label = spacy_labels[i]
        lm_token_indices = spacy2lm[i]
        if len(lm_token_indices) == 0:
            continue
        else:
            labels[lm_token_indices[0]] = label2id[label]

    return {'labels': list(labels)}


def apply_sliding_window(data, tokenizer, max_length, stride):
    input_ids = data['input_ids']  # without special tokens
    labels = data['labels']

    n_tokens = len(input_ids)
    if n_tokens <= (max_length - 2):
        n_window = 1
    else:
        if (n_tokens - (max_length - 2)) % (max_length - stride - 2) == 0:
            n_window = 1 + (n_tokens - (max_length - 2)) // (max_length - stride - 2)
        else:
            n_window = 2 + (n_tokens - (max_length - 2)) // (max_length - stride - 2)

    input_ids_window, labels_window = [], []
    for i in range(n_window):
        start_idx = i * (max_length -  stride - 2)
        end_idx = start_idx + max_length - 2

        input_ids_i = input_ids[start_idx:end_idx]
        labels_i = labels[start_idx:end_idx]

        # add special tokens
        input_ids_i.insert(0, tokenizer.vocab.get(tokenizer.special_tokens_map['bos_token']))
        input_ids_i.append(tokenizer.vocab.get(tokenizer.special_tokens_map['eos_token']))
        labels_i.insert(0, -100)
        labels_i.append(-100)

        input_ids_window.append(input_ids_i)
        labels_window.append(labels_i)

    # sanity check
    tokenized = tokenizer(
        data['full_text'],
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True
    )

    for i in range(n_window):
        assert input_ids_window[i] == tokenized['input_ids'][i]

    return {'input_ids': input_ids_window, 'labels': labels_window}


def preprocess_pipeline(data_list, tokenizer, config, label2id, is_train=True):
    if not is_train:
        for data in data_list:
            data['labels'] = [-1] * len(data['tokens'])

    dataset = Dataset.from_list(data_list)
    dataset = dataset.rename_columns({'tokens': 'spacy_tokens', 'labels': 'spacy_labels'})

    # token alignments
    dataset = dataset.map(lambda x: tokenizer(x['full_text'], return_length=True, add_special_tokens=False), num_proc=4)
    dataset = dataset.map(get_alignments, fn_kwargs={'tokenizer': tokenizer}, num_proc=4)

    # label alignment
    if is_train:
        dataset = dataset.map(align_labels, fn_kwargs={'label2id': label2id}, num_proc=4)
    else:
        dataset = dataset.map(lambda x: {'labels': [-1] * len(x['input_ids'])}, num_proc=4)

    # apply sliding window
    if is_train:
        dataset = dataset.map(
            apply_sliding_window,
            fn_kwargs={'tokenizer': tokenizer, 'max_length': config.max_token_length, 'stride': config.stride},
            num_proc=8
        )
    else:
        dataset = dataset.map(lambda x: tokenizer(
            x['full_text'],
            max_length=config.max_token_length,
            truncation=True,
            stride=config.stride,
            return_overflowing_tokens=True,
            )
        )

    return dataset
