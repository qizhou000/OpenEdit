from transformers import  AutoTokenizer


def set_tokenizer_pad_id(tokenizer:AutoTokenizer):
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print('Set [pad_token] as [eos_token].')
