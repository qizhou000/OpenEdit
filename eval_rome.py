from editors.rome import ROME, ROMEConfig
from transformers import  AutoTokenizer, AutoModelForCausalLM
from evaluate.evaluation import Evaluation
from utils.data import TestSampleList

def rome_evaluation_zsre(model_path, rome_stats_dir, rome_config_path = 'configs/rome/gpt2-xl.yaml', device = 'cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_path) 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # initialize rome
    config = ROMEConfig.from_yaml(rome_config_path)
    rome = ROME(model, tokenizer, config, rome_stats_dir, device, False)
    # Evaluation
    data_path = 'data/evaluation/zsre/zsre_mend_eval.json'
    test_sample_list = TestSampleList.zsre(data_path, None)
    ev = Evaluation(rome, test_sample_list, None)
    ev.evaluate_single_edit()
    ev.evaluate_sequential_edit(10)
    ev.evaluate_sequential_edit(100)
    ev.evaluate_sequential_edit(1000)

model_path = ... # gpt2-xl path
rome_stats_dir = ... # rome stats download from  https://rome.baulab.info/data/stats/
rome_evaluation_zsre(model_path, rome_stats_dir, rome_config_path = 'configs/rome/gpt2-xl.yaml', device = 'cuda')
