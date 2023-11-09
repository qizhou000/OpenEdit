#%%
import threading, time
import numpy as np
import torch, os, json, re
from typing import Dict, List, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from utils.utils import set_tokenizer_pad_id
from transformers import  AutoTokenizer 
from datasets import load_dataset
from queue import Queue 
 

################################################################################
# A Parallel Dataset class: Preprocessing and generating data batches through  #
# sub processes.                                                               #
################################################################################
class ParallelDataset():
    def __init__(self, sample_count:int, get_data_by_ids_func,
        batch_size = 256, random = True, random_batch = False, buffer_size = 64, 
        drop_last = False, random_seed = None) -> None:
        '''
        基本数据集类，若子类继承需实现`__get_data_by_ids__(self, ids)`函数，用于
        通过id获取数据。可以将根据id获取的数据的预处理都放在这个函数中进行，因为
        该类通过额外线程实现数据获取，并将获取的数据保存在队列中。
        sample_count: 样本总数
        get_data_by_ids_func: 通过id获取数据的函数。 
        batch_size: 生成数据批量大小
        random: 是否随机生成
        random_batch: 是否随机抽取不大于batch_size的批量大小
        buffer_size: 子进程保存数据的缓冲区大小 
        drop_last: 是否丢弃最后不成批量的数据，不丢弃的话就和下一个epoch的数据合在一块输出
        '''
        self.sample_count = sample_count
        self.batch_size = min(batch_size, sample_count)
        self.random_batch = random_batch
        self.random = random
        self.rng = np.random.default_rng(random_seed)
        self.select_ids = np.array(range(sample_count))
        if random:
            self.rng.shuffle(self.select_ids)
        self.drop_last = drop_last
        self.now_buffer_i = 0 # the idex of data has added into buffer
        self.now_yield_i = 0 # the idex of data has yielded
        self.buffer_size = buffer_size
        self.buffer = Queue()
        self.is_loading_data = False
        self.__get_data_by_ids__ = get_data_by_ids_func
        self.__fill_buffer__()

    def __get_data_by_ids__(self, ids):
        raise

    def __fill_buffer__(self):
        if self.is_loading_data:
            return
        def fill_buffer(): 
            self.is_loading_data = True 
            while self.buffer.qsize() < self.buffer_size:
                bs = self.batch_size
                if self.random_batch:
                    bs = self.rng.integers(1, self.batch_size+1)
                tail_i = self.now_buffer_i + bs
                ids = self.select_ids[self.now_buffer_i:tail_i]
                if tail_i >= self.sample_count:
                    self.select_ids = np.array(range(self.sample_count))
                    if self.random:
                        self.rng.shuffle(self.select_ids)
                    if tail_i > self.sample_count and self.drop_last:
                        self.now_buffer_i = 0
                        continue
                    self.now_buffer_i = tail_i - self.sample_count
                    extra_ids = self.select_ids[:self.now_buffer_i]
                    ids = np.concatenate([ids, extra_ids], 0)
                else:
                    self.now_buffer_i = tail_i
                d = self.__get_data_by_ids__(ids)
                self.buffer.put((d, len(ids)))
            self.is_loading_data = False  
        threading.Thread(target = fill_buffer).start() 
    
    def __len__(self): 
        batch_size = self.batch_size
        if self.random_batch:
            print('The number of data batches is not accurate as `random_batch` is set as True')
            batch_size = (self.batch_size + 1) / 2
        if self.drop_last:
            return int(np.floor(self.sample_count/batch_size))
        return int(np.ceil(self.sample_count/batch_size))

    def __iter__(self): 
        self.now_yield_i = 0
        return self

    def __next__(self):
        if self.now_yield_i >= self.sample_count:
            raise StopIteration
        if self.buffer.qsize() <= self.buffer_size/2:
            self.__fill_buffer__() 
        t = 0  
        while self.buffer.qsize() == 0:  
            print('\r', "Waiting data: %d s"%t, end='')
            time.sleep(1) 
            t += 1  
        d, data_n = self.buffer.get()
        self.now_yield_i += data_n
        return d


# def get_data_by_ids_func(ids):
#     return ids

# data = ParallelDataset(4, get_data_by_ids_func, 3, False, False, 32, False, 5)
# for i in range(5):
#     for j in data:
#         print(j)

################################################################################
#    prompts & targets transform to input&output&mask token ids                # 
################################################################################
def prompts_target_to_x_y_mask(tokenizer, prompts:List[str], targets:List[str], device='cuda'):
    '''
    生成训练自回归模型的输入x和输出y，以及训练mask。
    假设 prompts 与 targets 一一对应
    return input_ids, label_ids, masks
    input_ids/label_ids/masks's type, dtype, shape: 
        torch.Tensor, Long, [batch_size, max_length_of_prompts_and_targets]
    '''
    targets = deepcopy(targets)
    for i, t in enumerate(targets):
        targets[i] = t if t[0] == ' ' else ' ' + t
    input_ids, label_ids, masks = [], [], []
    for p, t in zip(prompts, targets):
        prompt_tok = tokenizer(p)['input_ids']
        input_tok = tokenizer(p + t, return_tensors="pt")['input_ids'][0]
        label_tok = input_tok.clone()[1:] 
        input_tok = input_tok[:-1] # 最后一个token没有下一个token
        mask = torch.ones_like(label_tok)
        mask[:len(prompt_tok)-1] *= 0
        input_ids.append(input_tok)
        label_ids.append(label_tok)
        masks.append(mask)
    input_ids = pad_sequence(input_ids, True, tokenizer.pad_token_id).to(device)
    label_ids = pad_sequence(label_ids, True, tokenizer.pad_token_id).to(device)
    masks = pad_sequence(masks, True, 0).to(device)
    return input_ids, label_ids, masks


################################################################################
#    prompts & predict length to get input&output&mask token ids               #  
################################################################################
def prompts_last_len_to_x_y_mask(tokenizer, prompts:List[str], pre_len:Union[int, float], 
    truncation = 1024, device='cuda'):
    '''
    将 prompt 的 token 最后 pre_len 数量/比例 的 token 用于预测输出，
    生成训练自回归模型的输入x和输出y，以及训练mask
    truncation: 为了防止token过多，省略多余的token
    input_ids/label_ids/masks's type, dtype, shape: 
        torch.Tensor, Long, [batch_size, max_length_of_prompts]
    '''
    input_ids, label_ids, masks = [], [], []
    for p in prompts:
        input_tok = tokenizer(p, return_tensors="pt")['input_ids'][0][:truncation]
        label_tok = input_tok.clone()[1:] 
        input_tok = input_tok[:-1] # 最后一个token没有下一个token
        mask = torch.zeros_like(label_tok)
        if type(pre_len) == int:
            mask[-pre_len:] += 1
        elif type(pre_len) == float and pre_len <= 1.:
            pl = int(len(mask) * pre_len)
            mask[-pl:] += 1
        else:
            raise
        input_ids.append(input_tok)
        label_ids.append(label_tok)
        masks.append(mask)
    input_ids = pad_sequence(input_ids, True, tokenizer.pad_token_id).to(device)
    label_ids = pad_sequence(label_ids, True, tokenizer.pad_token_id).to(device)
    masks = pad_sequence(masks, True, 0).to(device)
    return input_ids, label_ids, masks



################################################################################
#              Initialize for training datasets                                #
################################################################################
class TrainDataInit:
    '''
    Functions that preprocess the training datasets and output the `get_data_by_ids_func` 
    for `ParallelDataset` class to generating data.
    '''
    # ZSRE
    def zsre(data_path, tokenizer:AutoTokenizer, device='cuda'):
        assert os.path.isfile(data_path)
        set_tokenizer_pad_id(tokenizer)
        with open(data_path, 'r') as f: 
            data = json.load(f)
            sample_count = len(data)
            prompts = np.array([i['src'] for i in data])
            rep_prompts = np.array([i['rephrase'] for i in data])
            target_new = np.array([i['alt'] for i in data])
            loc_prompts = np.array([i['loc'] for i in data])
            loc_ans = np.array([i['loc_ans'] for i in data])
        def get_data_by_ids(ids:List[int]):
            edit_xym = prompts_target_to_x_y_mask(tokenizer, prompts[ids], target_new[ids], device)
            rep_xym = prompts_target_to_x_y_mask(tokenizer, rep_prompts[ids], target_new[ids], device)
            loc_xym = prompts_target_to_x_y_mask(tokenizer, loc_prompts[ids], loc_ans[ids], device)
            return edit_xym, {'rephrase': rep_xym}, {'normal': (loc_xym[0], loc_xym[2])}
        return sample_count, get_data_by_ids
    # WIKI
    def wiki(data_path, tokenizer:AutoTokenizer, data_type = 'train', 
        pre_len:Union[int, float] = 0.3, truncation = 1024, device='cuda'):
        '''
        if_train: 取训练集还是测试集
        pre_len: int 或 float，如果是int就取最后固定长度的token进行预测，如果是float，
            且小于等于1，就取最后占比为该比例的token进行预测
        '''
        assert os.path.isdir(data_path) 
        set_tokenizer_pad_id(tokenizer)
        ds = load_dataset(data_path)[data_type]
        # 去掉词数小于20且首尾是等号的串（章节名称）
        ds =  np.array([t for t in ds['text'] if len(t.split(' ')) > 20 and not \
            re.search(r'^[\s\n]*=', t) and not re.search(r'=[\s\n]*$', t)]) # 字符串列表
        def get_data_by_ids(ids):
            input_ids, label_ids, masks = prompts_last_len_to_x_y_mask(tokenizer, 
                ds[ids], pre_len, truncation, device)
            return input_ids, label_ids, masks
        sample_count = len(ds)
        return sample_count, get_data_by_ids


    

################################################################################
#                     get structured test datasets                             #
################################################################################
class TestSampleList:
    '''
    Functions used to read and preprocess various datasets for evaluation,
    which return list with structure like [
        { # test1
            'request': {'prompt': str, 'target_new': str, ...},
            'generality': {
                'gen_1_name':[
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ],
                'gen_2_name':[
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ], ...
            },
            'locality': {
                'loc_1_name':[
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ],
                'loc_2_name':[
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ], ...
            }
        }, 
        { # test2
            'request':{'prompt': str, 'target_new': str, ...},
            'generality': ...
        }, ...
    ]. 
    '''
    def zsre(path, test_i:Union[List, int] = None):
        test_sample_list = []
        with open(path, 'r') as f:
            data = json.load(f)
            if test_i == None:
                test_i = range(len(data))
            elif type(test_i) == int:
                test_i = range(test_i)
            elif type(test_i) != list:
                raise
            for s in [data[i] for i in test_i]:
                ns = {}
                ns['request'] = {
                    'prompt': s['src'], 
                    'target_new': s['alt'], 
                    'subject': s['subject']
                }
                ns['generality'] = {
                    'rephrase': [
                        {'prompt': s['rephrase'], 'target': s['alt']},
                    ]
                }
                ns['locality'] = {
                    'loc1': [
                        {'prompt': s['loc'], 'target': s['loc_ans']},
                    ]
                }
                test_sample_list.append(ns)
        return test_sample_list

    def counterfact(path):
        pass

    def test_data(path, test_n = None):
        with open(path, 'r') as f:
            data = json.load(f)
        return data










 