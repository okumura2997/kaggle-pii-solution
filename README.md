# Kaggle - The Learning Agency Lab - PII Data Detection: 13th place solution code

## Data preparation
Run the following command to download the competition data and external datasets:
```
mkdir input
kaggle competitions download -c pii-detection-removal-from-educational-data -p input
unzip input/pii-detection-removal-from-educational-data.zip -d input/pii-detection-removal-from-educational-data
kaggle datasets download -d mpware/pii-mixtral8x7b-generated-essays -p input
unzip input/pii-mixtral8x7b-generated-essays.zip -d input/pii-mixtral8x7b-generated-essays
kaggle datasets download -d nbroad/pii-dd-mistral-generated -p input
unzip input/pii-dd-mistral-generated.zip -d input/pii-dd-mistral-generated
rm input/*.zip
```

## Environments
Run the following command to run a container:
```
docker build -t pii-kaggle ./docker
docker run -it --detach --gpus all --shm-size 64G -v .:/kaggle --name pii-kaggle pii-kaggle /bin/bash
```

## Training
Run the following command to perform full-fit training:
```
python src/train.py exp019 --full-fit
```
The final ensemble consists of `exp019`, `exp021`, `exp034`, `exp037`, `exp038`.

## Inference
For inference, please refer to [submission code on kaggle](https://www.kaggle.com/code/yukiokumura1/pii-019-021-034-037-038-pp)