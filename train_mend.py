import torch
from torch import nn
from editors.mend import MEND, MENDConfig
from transformers import  AutoTokenizer, AutoModelForCausalLM
from utils.data import ParallelDataset, TrainDataInit

def train_mend_zsre(model_path, mend_config_path = 'configs/mend/gpt2-xl.yaml',
                    mend_ckpt_path = None, train_batch_size = 10, device = 'cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_path) 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = MENDConfig.from_yaml(mend_config_path)
    mend = MEND(model, tokenizer, config, device)
    mend.train_init('train_records', None, mend_ckpt_path, 1000, 10, 999, 999)
    # prepare dataset
    data_path = 'data/meta-train/zsre/zsre_mend_train.json'
    sample_count, get_data_by_ids = TrainDataInit.zsre(data_path, tokenizer, device)
    md = ParallelDataset(sample_count, get_data_by_ids, train_batch_size, True)
    # train mend
    mend.train(1000, md)

 
model_path = ... # gpt2-xl path
mend_ckpt_path = None # train from scratch
train_mend_zsre(model_path, mend_config_path = 'configs/mend/gpt2-xl.yaml',
                    mend_ckpt_path = None, train_batch_size = 10, device = 'cuda')
