from types import SimpleNamespace
from transformers import TrainingArguments


config = SimpleNamespace(
    model_path='microsoft/deberta-v3-large',
    seed=42,
    max_token_length=1024,
    stride=256,
    freeze_embeddings=True,
    freeze_layers=6,
    loss='focal',
    external_data_paths=[
        '/kaggle/input/fix-punctuation-tokenization-external-dataset/pii_dataset_fixed.json',  # moth
        '/kaggle/input/fix-punctuation-tokenization-external-dataset/moredata_dataset_fixed.json',  # pj
        '/kaggle/input/pii-dd-mistral-generated/mixtral-8x7b-v1.json',  # nbroad
        '/kaggle/input/pii-mixtral8x7b-generated-essays/mpware_mixtral8x7b_v1.1.json',  # mpware
        '/kaggle/input/mixtral-original-prompt/Fake_data_1850_218.json'  # nofit
    ],
)

train_args = TrainingArguments(
    output_dir='./',
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    report_to='wandb',
    evaluation_strategy='no',
    do_eval=False,
    logging_strategy='steps',
    logging_steps=20,
    save_strategy='epoch',
    save_total_limit=2,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    weight_decay=1e-2,
    fp16=True,
)
