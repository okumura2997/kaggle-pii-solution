FROM nvcr.io/nvidia/pytorch:24.01-py3
WORKDIR /kaggle

RUN pip install --no-cache-dir \
    wandb==0.16.3 \
    transformers[torch]==4.37.2 \
    datasets==2.17.0 \
    evaluate==0.4.1 \
    seqeval==1.2.2 \
    spacy==3.7.2 \
    spacy-alignments==0.9.1 \
    sentencepiece==0.1.99 \
    seaborn==0.13.2