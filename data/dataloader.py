from recbole.data.dataloader.general_dataloader import TrainDataLoader, FullSortEvalDataLoader

from data.transform import construct_transform


class CustomizedTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.transform = construct_transform(config)

    def _next_batch_data(self):
        cur_data = super()._next_batch_data()
        transformed_data = self.transform(self, cur_data)
        return transformed_data


class CustomizedFullSortEvalDataLoader(FullSortEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.transform = construct_transform(config)

    def _next_batch_data(self):
        cur_data = super()._next_batch_data()
        transformed_data = self.transform(self, cur_data[0])
        return (transformed_data, *cur_data[1:])
