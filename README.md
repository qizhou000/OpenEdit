# OpenEdit
A Modular Language Model Editing Repository easy to call and evaluate 
model editing methods on diverse language models.
New editor code is being added including MEMIT, SERAC, T-Patcher, etc.

# DEMO
To editing a language model using specific editing method, you just need to perform three steps. The following shows the three steps for evaluating ROME:
1. Instantiate a language model to be edited and its corresponding tokenizer.
```
from transformers import  AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path) 
tokenizer = AutoTokenizer.from_pretrained(model_path)
```
2. Initialize a model editor, pass in the instantiated model and tokenizer, as well as the corresponding editor configuration,
where directory path `rome_stats_dir` is the statistical matrix specifically required for the ROME editing method. You can download for GPT2-XL and GPT-J-6B from official website
<https://rome.baulab.info/data/stats/>. If set `rome_stats_dir = None`, the program will download the wiki dataset and compute the statistical matrix automatically.
```
from editors.rome import ROME, ROMEConfig
config = ROMEConfig.from_yaml(rome_config_path)
rome = ROME(model, tokenizer, config, rome_stats_dir)
```
3. 



Python script `eval_rome.py` shows a demonstration of editing GPT2-XL using ROME.

# Extra Editor
If you want to implement a new language model editor, please inherit the base 
class `editors.editor.BaseEditor` and implement the corresponding abstract functions.


# Extra Evaluation Dataset
If you want to evaluate editors on a new dataset, please organize the data 
structure to match the argument `test_sample_list` passed to the 
`evaluate.evaluation.Evaluation` class.

