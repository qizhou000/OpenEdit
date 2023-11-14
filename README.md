# OpenEdit
A Modular Language Model Editing Repository easy to call and evaluate 
model editing methods on diverse language models.
New editor code is being added including MEMIT, SERAC, T-Patcher, etc.



# DEMO
## EVALUATION
To EVALUATING an editing method on specific language model, you just need to perform three steps. The following shows the three steps for evaluating ROME:
1. Instantiate a language model to be edited and its corresponding tokenizer.
```python
from transformers import  AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path) 
tokenizer = AutoTokenizer.from_pretrained(model_path)
```
2. Initialize a model editor, pass in the instantiated model and tokenizer, as well as the corresponding editor configuration,
where directory path `rome_stats_dir` is the statistical matrix specifically required for the ROME editing method. You can download for GPT2-XL and GPT-J-6B from official website
<https://rome.baulab.info/data/stats/>. If set `rome_stats_dir = None`, the program will download the wiki dataset and compute the statistical matrix automatically.
```python
from editors.rome import ROME, ROMEConfig
config = ROMEConfig.from_yaml(rome_config_path)
rome = ROME(model, tokenizer, config, rome_stats_dir)
```
3. Evaluate.
```python
data_path = 'data/evaluation/zsre/zsre_mend_eval.json'
test_sample_list = TestSampleList.zsre(data_path, None)
ev = Evaluation(rome, test_sample_list, None)
ev.evaluate_single_edit()
ev.evaluate_sequential_edit(10)
ev.evaluate_sequential_edit(100)
ev.evaluate_sequential_edit(1000)
```
The python script `eval_rome.py` executed the above code. 

## EDITING
If you simply want to use one or a few samples to edit the model and perform other subsequent operations, you can run:
```python
request = { # for example
    'prompt': 'The Space Needle is located in',
    'subject': 'The Space Needle',
    'target_new': " London"
}
eome_editor.edit_one_piece(request) 
```
for one sample, or 
```python
requests = [{ # for example
    'prompt': 'The Space Needle is located in',
    'subject': 'The Space Needle',
    'target_new': " London"
}]
eome_editor.edit_batch(requests) 
```
for batched samples (only supported by a few editors). If you want to restore the edited model to the original model, run:
```python
eome_editor.restore_to_original_model(request) # 
```


# Extra Editor
If you want to implement a new language model editor, please inherit the base 
class `editors.editor.BaseEditor` and implement the corresponding abstract functions.


# Extra Evaluation Dataset
If you want to evaluate editors on a new dataset, please organize the data 
structure to match the argument `test_sample_list` passed to the 
`evaluate.evaluation.Evaluation` class.

