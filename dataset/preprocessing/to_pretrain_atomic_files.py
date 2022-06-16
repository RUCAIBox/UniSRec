import os
import argparse

from utils import check_path


def load_atomic_file(dataset_id, parent_data, train_data_path):
    with open(train_data_path, 'r', encoding='utf-8') as file:
        file.readline()
        for line in file:
            uid, item_seq, target_iid = line.strip().split('\t')
            item_seq = item_seq.split(' ')
            uid = f'{dataset_id}-' + uid
            item_seq = [f'{dataset_id}-' + _ for _ in item_seq]
            target_iid = f'{dataset_id}-' + target_iid
            parent_data.append([uid, item_seq, target_iid])


def merge_and_save(dataset_names, input_path, output_path):
    dataset_names = dataset_names.split(',')
    print('Convert dataset: ')
    print(' Dataset: ', dataset_names)

    train_data = []
    valid_data = []
    test_data = []
    for i, dataset_name in enumerate(dataset_names):
        print(i, dataset_name)
        load_atomic_file(i, train_data,
            os.path.join(input_path, dataset_name, f'{dataset_name}.train.inter'))
        load_atomic_file(i, valid_data,
            os.path.join(input_path, dataset_name, f'{dataset_name}.valid.inter'))
        load_atomic_file(i, test_data,
            os.path.join(input_path, dataset_name, f'{dataset_name}.test.inter'))

    uid_list = list({_[0] for _ in train_data})
    def cmp(t):
        base, value = t.split('-')
        return int(base) * 10000000 + int(value)
    uid_list.sort(key=cmp)

    short_name = ''.join([_[0] for _ in dataset_names])
    check_path(os.path.join(output_path, short_name))

    for token, merged_data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
        with open(os.path.join(output_path, short_name, f'{short_name}.{token}.inter'), 'w') as file:
            file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for line in merged_data:
                uid, item_seq, target_iid = line
                file.write(f'{uid}\t{" ".join(item_seq)}\t{target_iid}\n')

    with open(os.path.join(output_path, short_name, f'{short_name}.pt_datasets'), 'w') as file:
        file.write(','.join(dataset_names) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='Food,Home,CDs,Kindle,Movies',
                        help='Combination of pre-trained datasets, split by comma')
    parser.add_argument('--input_path', type=str, default='../pretrain/')
    parser.add_argument('--output_path', type=str, default='../pretrain/')
    args = parser.parse_args()

    merge_and_save(args.datasets, args.input_path, args.output_path)
