import argparse
import csv
import datetime
import os
from tqdm import tqdm

from process_amazon import filter_inters, make_inters_in_order, get_user_item_from_ratings, \
                           generate_training_data, generate_item_embedding, convert_to_atomic_files
from utils import check_path, set_device, load_plm


def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        fp.readline()
        cr = csv.reader(fp)
        for line in tqdm(cr, desc='Load ratings'):
            try:
                InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country = line
                if Description == '':
                    continue
                users.add(InvoiceNo)
                items.add(StockCode)
                ts = datetime.datetime.strptime(InvoiceDate, '%m/%d/%Y %H:%M').timestamp()
                inters.add((InvoiceNo, StockCode, float(Quantity), int(ts)))
            except ValueError:
                print(line)
    return users, items, inters


def preprocess_rating(args):
    print('Process rating data: ')
    print(' Dataset: ', args.dataset)

    # load ratings
    rating_file_path = os.path.join(args.input_path, 'data-utf8.csv')
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters: ', len(rating_inters))
    rating_inters = filter_inters(rating_inters, can_items=rating_items,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    return rating_inters


def generate_text(args, items):
    item_text_list = []

    meta_file_path = os.path.join(args.input_path, 'data-utf8.csv')
    item2text = {}
    with open(meta_file_path, 'r') as fp:
        fp.readline()
        cr = csv.reader(fp)
        for line in tqdm(cr, desc='Load ratings'):
            try:
                InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country = line
                if Description == '':
                    continue
                if StockCode not in item2text:
                    item2text[StockCode] = Description
            except ValueError:
                print(line)

    for iid in tqdm(items, desc='Generate text'):
        assert iid in item2text
        text = item2text[iid].strip().lower() + '.'
        item_text_list.append([iid, text])
    return item_text_list


def preprocess_text(args, rating_inters):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)
    rating_users, rating_items = get_user_item_from_ratings(rating_inters)

    # load item text and clean
    item_text_list = generate_text(args, rating_items)
    print('\n')

    # return: list of (item_ID, cleaned_item_text)
    return item_text_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='OR')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='../raw/')
    parser.add_argument('--output_path', type=str, default='../downstream/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load interactions from raw rating file
    rating_inters = preprocess_rating(args)

    # load item text from raw meta data file
    item_text_list = preprocess_text(args, rating_inters)

    # split train/valid/test
    train_inters, valid_inters, test_inters, user2index, item2index = \
        generate_training_data(args, rating_inters)

    # device & plm initialization
    device = set_device(args.gpu_id)
    args.device = device
    plm_tokenizer, plm_model = load_plm(args.plm_name)
    plm_model = plm_model.to(device)

    # create output dir
    check_path(os.path.join(args.output_path, args.dataset))

    # generate PLM emb and save to file
    generate_item_embedding(args, item_text_list, item2index, 
                            plm_tokenizer, plm_model, word_drop_ratio=-1)
    # pre-stored word drop PLM embs
    if args.word_drop_ratio > 0:
        generate_item_embedding(args, item_text_list, item2index, 
                                plm_tokenizer, plm_model, word_drop_ratio=args.word_drop_ratio)

    # save interaction sequences into atomic files
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)
