import os
import torch
from recbole.config import Config as RecBoleConfig

class Config(RecBoleConfig):
    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu and ('ddp' in self.final_config_dict and not self.final_config_dict['ddp']):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
