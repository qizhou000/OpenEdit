import torch
from torch import nn
from editors.mend import MEND, MENDConfig
from transformers import  AutoModelForCausalLM, AutoModelForCausalLM
from evaluate.evaluation import Evaluation
from utils.data import TestSampleList

def mend_evaluation_zsre(model_path, mend_ckpt_path, mend_config_path = 'configs/mend/gpt2-xl.yaml', device = 'cuda'):
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_path) 
    tokenizer = AutoModelForCausalLM.from_pretrained(model_path)
    config = MENDConfig.from_yaml(mend_config_path)
    # load mend auxiliary model
    mend = MEND(model, tokenizer, config, device)
    mend.load_ckpt(mend_ckpt_path)
    # dataset
    data_path = 'data/evaluation/zsre/zsre_mend_eval.json'
    test_sample_list = TestSampleList.zsre(data_path, None)
    # Evaluate
    ev = Evaluation(mend, test_sample_list, None)
    ev.evaluate_single_edit()
    ev.evaluate_batch_edit(batch_size = 64)
    ev.evaluate_sequential_edit(sequential_edit_n = 10)
    ev.evaluate_sequential_edit(sequential_edit_n = 100)
    ev.evaluate_sequential_edit(sequential_edit_n = 1000)

model_path = ... # gpt2-xl path
mend_ckpt_path = ... 
mend_evaluation_zsre(model_path, mend_ckpt_path, mend_config_path = 'configs/mend/gpt2-xl.yaml', device = 'cuda')
