from types import SimpleNamespace
from transformers import TrainingArguments


config = SimpleNamespace(
    model_path='/kaggle/outputs/pretrain002/checkpoint-5325',
    seed=42,
    n_folds=4,
    folds=[0, 1, 2, 3],
    max_token_length=1024,
    stride=256,
    freeze_embeddings=True,
    freeze_layers=12,
    o_thresholds=[0.85, 0.90, 0.95],
    loss='focal',
    external_data_paths=[
        # '/kaggle/input/fix-punctuation-tokenization-external-dataset/pii_dataset_fixed.json',  # moth
        # '/kaggle/input/fix-punctuation-tokenization-external-dataset/moredata_dataset_fixed.json',  # pj
        # '/kaggle/input/pii-dd-mistral-generated/mixtral-8x7b-v1.json',  # nbroad
        # '/kaggle/input/pii-mixtral8x7b-generated-essays/mpware_mixtral8x7b_v1.1.json',  # mpware
        # '/kaggle/input/mixtral-original-prompt/Fake_data_1850_218.json'  # nofit
    ],
)

if hasattr(config, 'sampling_weights'):
    assert sum(config.sampling_weights) == 1
    assert len(config.external_data_paths) + 1 == len(config.sampling_weights)

train_args = TrainingArguments(
    output_dir='./',
    learning_rate=1e-5,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    report_to='wandb',
    evaluation_strategy='steps',
    eval_steps=0.05,
    logging_strategy='steps',
    logging_steps=0.01,
    save_strategy='steps',
    save_steps=0.05,
    save_total_limit=1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    weight_decay=1e-2,
    metric_for_best_model='loss',
    fp16=True,
    load_best_model_at_end=True,
)
