import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from recbole.quick_start import run_recbole

from data.dataset import UniSRecDataset


def run_baseline(model, dataset, config_file_list=[]):
    # configurations initialization
    model_name = model
    if f'props/{model_name}.yaml' not in config_file_list:
        config_file_list = [f'props/{model_name}.yaml'] + config_file_list
    print(config_file_list)

    # configurations initialization
    config = Config(model=model_name, dataset=dataset, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = UniSRecDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    if model_name == 'FDSA':
        from baselines.fdsa import FDSA
        model = FDSA(config, train_data.dataset).to(config['device'])
    elif model_name == 'S3Rec':
        from baselines.s3rec import S3Rec
        model = S3Rec(config, train_data.dataset).to(config['device'])
    else:
        raise NotImplementedError(f'The baseline [{model_name}] has not implemented yet.')
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Scientific', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='props/finetune.yaml', help='config files')

    args, _ = parser.parse_known_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    if args.model in ['FDSA', 'S3Rec']:
        baseline_func = run_baseline
    else:
        baseline_func = run_recbole

    baseline_func(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
