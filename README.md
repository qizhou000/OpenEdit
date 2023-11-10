# OpenEdit
A Modular Language Model Editing Repository that is easy to call and evaluate 
model editing methods on diverse language models.
New editor code is being added including MEMIT, SERAC, T-Patcher, etc.

# Evaluation
If you want to evaluate editors, use `evaluate.evaluation.Evaluation`.

# Extra Editor
If you want to implement a new language model editor, please inherit the base 
class `editors.editor.BaseEditor` and implement the corresponding abstract functions.


# Extra Evaluation Dataset
If you want to evaluate editors on a new dataset, please organize the data 
structure to match the argument `test_sample_list` passed to the 
`evaluate.evaluation.Evaluation` class.

