Please download the processed datasets from [Google Drive](https://drive.google.com/drive/folders/1Uik0fMk4oquV_bS9lXTZuExAYbIDkEMW?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1zdP3tEw9X6Ys5YNO5TyNEQ) (密码 3cml), and move them here.

```
dataset/
  pretrain/
    FHCKM/
  downstream/
    Scientific/
    Pantry/
    Instruments/
    Arts/
    Office/
    OR/     # Online Retail
```

# Dataset Preprocessing

If you have downloaded the processed datasets, you can directly use them for reproduction and further experiments.

If you want to know the details of data preprocessing, please see the instructions below.

## Amazon 2018

### 1. Download raw datasets

Please download the raw datasets from the original website [[link]](https://nijianmo.github.io/amazon/index.html).

Here we take `Pantry` for example.

```
dataset/
  raw/
    Metadata/
      meta_Prime_Pantry.json.gz
    Ratings/
      Prime_Pantry.csv
```

### 2. Process downstream datasets

```bash
cd dataset/preprocessing/
python process_amazon.py --dataset Pantry
```
