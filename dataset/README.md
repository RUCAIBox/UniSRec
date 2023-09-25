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

> **[Important!!!]** Note that due to the issue of randomness, the processed datasets may not be exactly the same as those released by us. As some items are reviewed at the same timestamp, then these items can have a random order in the item sequences after sorting chronologically.

## Amazon 2018

### 1. Download raw datasets

Please download the raw datasets from the original website.

For the meta data, please click the ***metadata*** link of each category in the table "Complete review data" from [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html).

For the rating data, please click the ***ratings only*** link of each category in the table "Small subsets for experimentation" from [https://nijianmo.github.io/amazon/index.html#subsets](https://nijianmo.github.io/amazon/index.html#subsets).

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

### 3. Process pretrain datasets

```bash
# cd dataset/preprocessing/

for ds in Food Home CDs Kindle Movies
do
  python process_amazon.py --dataset ${ds} --output_path ../pretrain/ --word_drop_ratio 0.15
done

python to_pretrain_atomic_files.py

path=`pwd`
for ds in Food Home CDs Kindle Movies
do
  ln -s ${path}/../pretrain/${ds}/${ds}.feat1CLS ../pretrain/FHCKM/
  ln -s ${path}/../pretrain/${ds}/${ds}.feat2CLS ../pretrain/FHCKM/
done
```

## Online Retail

### 1. Download raw datasets

Please download the raw datasets from Kaggle [[link]](https://www.kaggle.com/datasets/carrie1/ecommerce-data) and save `archive.zip` into `dataset/raw/`.

Unzip and convert it to UTF-8.

```bash
mv archive.zip dataset/raw/
cd dataset/raw/
unzip archive.zip
iconv -f latin1 -t utf-8 data.csv > data-utf8.csv
```

### 2. Process downstream dataset

```bash
cd dataset/preprocessing/
python process_or.py
```

# Useful Files

You may find some files useful for your research, including:
  1. Clean item text (`*.text`);
  2. Index mapping between raw IDs and remapped IDs (`*.user2index`, `*.item2index`);

For downstream datasets, the corresponding files are naturally in the `downstream-datasets.zip`. Once you unzip it, then you may find them.

For pre-trained datasets, the corresponding files are stored in `raw-datasets-for-pretrain.zip`.
