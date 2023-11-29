from logging import getLogger
import random
import torch
from recbole.data.interaction import Interaction


def construct_transform(config):
    if config['unisrec_transform'] is None:
        logger = getLogger()
        logger.warning('Equal transform')
        return Equal(config)
    else:
        str2transform = {
            'plm_emb': PLMEmb
        }
        return str2transform[config['unisrec_transform']](config)


class Equal:
    def __init__(self, config):
        pass

    def __call__(self, dataset, interaction):
        return interaction


class PLMEmb:
    def __init__(self, config):
        self.logger = getLogger()
        self.logger.info('PLM Embedding Transform in DataLoader.')
        self.item_drop_ratio = config['item_drop_ratio']
        self.item_drop_coefficient = config['item_drop_coefficient']

    def __call__(self, dataset, interaction):
        '''Sequence augmentation and PLM embedding fetching
        '''
        item_seq_len = interaction['item_length']
        item_seq = interaction['item_id_list']

        plm_embedding = dataset.plm_embedding
        item_emb_seq = plm_embedding(item_seq)
        pos_item_id = interaction['item_id']
        pos_item_emb = plm_embedding(pos_item_id)

        mask_p = torch.full_like(item_seq, 1 - self.item_drop_ratio, dtype=torch.float)
        mask = torch.bernoulli(mask_p).to(torch.bool)

        # Augmentation
        rd = random.random()
        if rd < self.item_drop_coefficient:
            # Item drop
            seq_mask = item_seq.eq(0).to(torch.bool)
            mask = torch.logical_or(mask, seq_mask)
            mask[:,0] = True
            drop_index = torch.cumsum(mask, dim=1) - 1

            item_seq_aug = torch.zeros_like(item_seq).scatter(dim=-1, index=drop_index, src=item_seq)
            item_seq_len_aug = torch.gather(drop_index, 1, (item_seq_len - 1).unsqueeze(1)).squeeze() + 1
            item_emb_seq_aug = plm_embedding(item_seq_aug)
        else:
            # Word drop
            plm_embedding_aug = dataset.plm_embedding_aug
            full_item_emb_seq_aug = plm_embedding_aug(item_seq)

            item_seq_aug = item_seq
            item_seq_len_aug = item_seq_len
            item_emb_seq_aug = torch.where(mask.unsqueeze(-1), item_emb_seq, full_item_emb_seq_aug)

        interaction.update(Interaction({
            'item_emb_list': item_emb_seq,
            'pos_item_emb': pos_item_emb,
            'item_id_list_aug': item_seq_aug,
            'item_length_aug': item_seq_len_aug,
            'item_emb_list_aug': item_emb_seq_aug,
        }))

        return interaction
