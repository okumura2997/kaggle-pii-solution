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
    eval_batch_size=8,
    scheduler_type='cosine',
    warm_up_ratio=0.1,
    weight_decay=1e-2,
    freeze_embeddings=True,
    freeze_layers=6,
    evaluation_strategy='steps',
    eval_steps=100,
    save_steps=100,
    o_thresholds=[0.90, 0.95, 0.99],
    metric_for_best_model='ents_f5_0.95',

    external_data_paths=[
        # '/kaggle/input/fix-punctuation-tokenization-external-dataset/pii_dataset_fixed.json',  # moth
        # '/kaggle/input/fix-punctuation-tokenization-external-dataset/moredata_dataset_fixed.json',  # pj
        '/kaggle/input/pii-dd-mistral-generated/mixtral-8x7b-v1.json',  # nbroad
        '/kaggle/input/pii-mixtral8x7b-generated-essays/mpware_mixtral8x7b_v1.1.json',  # mpware
        # '/kaggle/input/mixtral-original-prompt/Fake_data_1850_218.json'  # nofit
    ],
    sampling_weights=[0.7, 0.15, 0.15]  # competition dataset, external dataset
)

if hasattr(config, 'sampling_weights'):
    assert sum(config.sampling_weights) == 1
    assert len(config.external_data_paths) + 1 == len(config.sampling_weights)
