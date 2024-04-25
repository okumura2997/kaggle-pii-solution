from types import SimpleNamespace

config = SimpleNamespace(
    model_path='microsoft/deberta-v3-small',
    seed=42,
    n_folds=4,
    folds=[0],
    max_token_length=1024,
    stride=256,
    lr=2e-5,
    n_epochs=1,
    train_batch_size=8,
    gradient_accumulation_steps=1,
    eval_batch_size=8,
    scheduler_type='cosine',
    warm_up_ratio=0.1,
    weight_decay=1e-2,
    freeze_embeddings=False,
    freeze_layers=0,
    evaluation_strategy='steps',
    eval_steps=100,
    save_steps=100,
    o_thresholds=[0.90, 0.95, 0.99],
    metric_for_best_model='ents_f5_0.95',
    external_data_paths=['/kaggle/input/pii-dd-mistral-generated/mixtral-8x7b-v1.json']
)