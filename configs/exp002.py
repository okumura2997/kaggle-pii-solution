from types import SimpleNamespace

config = SimpleNamespace(
    model_path='microsoft/deberta-v3-large',
    seed=42,
    n_folds=4,
    folds=[0, 1, 2, 3],
    max_token_length=1024,
    stride=256,
    lr=2e-5,
    n_epochs=3,
    train_batch_size=2,
    gradient_accumulation_steps=4,
    eval_batch_size=4,
    scheduler_type='cosine',
    warm_up_ratio=0.1,
    weight_decay=1e-2,
    freeze_embeddings=True,
    freeze_layers=6,
    o_threshold=0.90,
)