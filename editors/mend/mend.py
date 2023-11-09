import torch
import torch.nn as nn
import transformers
from ..editor import BaseEditor, EditorConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from .auxiliary_networks import GradientTransform
import json, yaml
from utils.data import prompts_target_to_x_y_mask
from torch.optim import Adam
from utils.data import ParallelDataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
import os
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

@dataclass
class MENDConfig(EditorConfig):
    @dataclass
    class AuxModelConfig():
        n_hidden: int
        hidden_dim: int
        init: str
        norm: bool
        act: str
        rank: int
        shared: bool
        lr: float
    edit_modules: List[str]
    if_edit_bias: bool
    init_edit_lr: float
    edit_lr_lr: float
    aux_model: AuxModelConfig
    edit_model_name: str
    relia_lambda: float
    gen_lambda: float
    loc_lambda: float

    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['aux_model'] = self.AuxModelConfig(**data['aux_model'])
        return self(**data)
    @classmethod
    def from_json(self, fpath):
        raise


class EditLinear():
    def __init__(self, edit_module:nn.Linear, module_name:str, idx:int, aux_model_weight:GradientTransform, 
                 config:MENDConfig, aux_model_bias:GradientTransform = None) -> None:
        '''
        创建待编辑模块（线性层），并为该模块注册输入与梯度的hook函数，每次原始待编辑
        模型前向、反向传播后，可使用 update_delta_weight 修改该模块后续的运算。
        idx: 模块在其weight形状的辅助网络的输入编号
        # 包含的继承nn.Module的模块/参数有：
        #     self.edit_module
        #     self.lr
        #     self.aux_model_weight
        #     self.aux_model_bias
        '''
        assert type(edit_module) == transformers.pytorch_utils.Conv1D or\
                type(edit_module) == nn.Linear
        self.edit_module = edit_module
        self.original_weight = self.edit_module.weight.clone()
        self.original_bias = self.edit_module.bias.clone()
        self.module_name = module_name
        self.idx = idx
        self.aux_model_weight = aux_model_weight
        self.if_edit_bias = config.if_edit_bias
        self.aux_model_bias = aux_model_bias
        self.lr = nn.Parameter(torch.tensor(config.init_edit_lr))
        self.register_hooks()
        self.clear_delta(True, True, True)

    def register_hooks(self):
        def forward_x_hook(module, args, output):
            # args: tuple(input) [batch_size, max_length, dim_in]
            # output.shape = [batch_size, max_length, dim_out]
            self.__x__ = args[0].detach()
        def backward_delta_hook(module, grad_input, grad_output):
            # grad_input: tuple(input grad) [batch_size, max_length, dim_in]
            # grad_output: tuple(output grad) [batch_size, max_length, dim_out]
            self.__delta__ = grad_output[0].detach()
        def forward_edit_hook(module, args, output):
            # args: tuple(input) [batch_size, max_length, dim_in]
            # output.shape = [batch_size, max_length, dim_out]
            if self.__delta_weight__ != None:
                output = output + args[0] @ self.__delta_weight__ # Wx + b + W'x = (W + W')x + b
            if self.__delta_bias__ != None:
                output = output + self.__delta_bias__ # Wx + b + W'x + b'
            return output
        if not hasattr(self, 'x_handle') or self.x_handle == None:
            self.x_handle = self.edit_module.register_forward_hook(forward_x_hook)
        if not hasattr(self, 'delta_handle') or self.delta_handle == None:
            self.delta_handle = self.edit_module.register_full_backward_hook(backward_delta_hook)
        if not hasattr(self, 'edit_handle') or self.edit_handle == None:
            self.edit_handle = self.edit_module.register_forward_hook(forward_edit_hook)
     
    # def remove_hooks(self, x_handle:bool, delta_handle:bool, edit_handle:bool):
    #     if x_handle:
    #         self.x_handle.remove()
    #         self.x_handle = None
    #     if delta_handle:
    #         self.delta_handle.remove()
    #         self.delta_handle = None
    #     if edit_handle:
    #         self.edit_handle.remove()
    #         self.edit_handle = None

    def update_delta_weight(self):
        '''
        Compute the weights updates using forward/backward and auxiliary model,
        which will influence model inference in hook function.
        '''
        assert self.__x__ != None and self.__delta__ != None
        # x: [request_num, dim_in], delta:[request_num, dim_out]
        x, delta = self.aux_model_weight(self.__x__, self.__delta__, self.idx) 
        if self.__delta_weight__ == None:
            self.__delta_weight__ = x.permute(1, 0) @ delta * self.lr # [dim_in, dim_out]
        else:
            self.__delta_weight__ += x.permute(1, 0) @ delta * self.lr
        if self.if_edit_bias:
            if self.aux_model_bias != None:
                delta = self.aux_model_weight(self.__delta__, self.idx)
            if self.__delta_bias__ == None:
                self.__delta_bias__ = delta.sum(0) * self.lr # [dim_out]
            else:
                self.__delta_bias__ += delta.sum(0) * self.lr
            self.__delta_bias__.reshape(1, 1, -1) 

    def clear_delta(self, x:bool, delta:bool, edit_weight:bool):
        if x:
            self.__x__ = None
        if delta:
            self.__delta__ = None
        if edit_weight:
            self.__delta_weight__ = None
            self.__delta_bias__ = None

    def update_weight_and_clear_delta(self):
        '''
        Add the delta weights into the weights of module. Then clear the delta 
        weights.
        '''
        if self.__delta_weight__.shape == self.edit_module.weight.shape:
            self.edit_module.weight += self.__delta_weight__.detach()
        else:
            self.edit_module.weight += self.__delta_weight__.permute(1, 0).detach()
        if self.if_edit_bias:
            self.edit_module.bias += self.__delta_bias__.flatten().detach()
        self.clear_delta(False, False, True)

    def restore_to_original_weight(self):
        self.edit_module.load_state_dict({'weight': self.original_weight,
                                          'bias': self.original_bias})

class MEND(BaseEditor):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
        config: MENDConfig, device = 'cuda'):

        super().__init__(model, tokenizer, device)
        self.cfg = config
        self.edit_modules, self.aux_models, self.autograd_params = self.get_edit_modules()
        self.aux_models = self.aux_models.to(device)
        self.edit_lrs = nn.ParameterList([em.lr for em in self.edit_modules]).to(device)
        self.log_writer = None
        self.set_train(False)
    
    ############################################################################
    #         Implementation Virtual Functions of Base Class                   #
    ############################################################################
    def name_of_editor_and_model(self):
        return 'mend', self.cfg.edit_model_name

    def if_can_batch_edit(self)->bool:
        return True

    def restore_to_original_model(self):
        self.clear_module_deltas(True, True, True)
        self.restore_to_original_weights()

    def edit_one_piece(self, request: Dict):
        '''request = {'prompt': str, 'target_new': str}'''
        self.edit_batch([request])

    def edit_batch(self, requests: List[Dict]):
        '''
        只在推理时使用，假设self.model为自回归模型
        requests = [
          {'prompt': str, 'target_new': str},
          {'prompt': str, 'target_new': str},
        ]
        '''
        prompts, new_targets = [], []
        for r in requests: 
            prompts.append(r['prompt'])
            new_targets.append(r['target_new'])
        input_ids, label_ids, masks = prompts_target_to_x_y_mask(self.tokenizer, prompts, new_targets, self.device)
        self.__edit_batch__(input_ids, label_ids, masks)
    
    def __edit_batch__(self, input_ids:torch.Tensor, label_ids:torch.Tensor, masks:torch.Tensor):
        # input_ids/label_ids/masks: [batch, max_len]
        self.autograd_params.requires_grad_(True)
        edit_loss = label_loss(self.model, input_ids, label_ids, masks)
        torch.autograd.grad(edit_loss, self.autograd_params)
        self.autograd_params.zero_grad()
        # 使用获取的__x__和__delta__计算权重更新，并用hook来修改后续模型的推理
        for em in self.edit_modules:
            em.update_delta_weight() 
        self.autograd_params.requires_grad_(False)

    ############################################################################
    #                   MEND Special Function                                  #
    ############################################################################
    def get_edit_modules(self) -> Tuple[List[EditLinear], nn.ModuleDict, nn.ParameterList]:
        '''
        获取模型中待编辑模块对象、辅助网络对象、用于自动梯度能hook到待编辑权重
        矩阵的梯度的模型中的参数
        '''
        same_shape_modules = defaultdict(list)
        for module_name in self.cfg.edit_modules:
            module = find_module(self.model, module_name.split('.'))
            if isinstance(module, transformers.pytorch_utils.Conv1D):
                in_dim, out_dim = module.weight.shape
            elif isinstance(module, nn.Linear):
                out_dim, in_dim = module.weight.shape
            else:
                raise 'Modified module should be liear.'
            shape = (in_dim, out_dim)
            same_shape_modules[shape].append([module, module_name])
        edit_modules, aux_models, autograd_params = [], nn.ModuleDict(), nn.ParameterList()
        for shape, modules in same_shape_modules.items():
            aux_model = GradientTransform(shape[0], shape[1], self.cfg.aux_model, len(modules))
            aux_models[str(shape)] = aux_model
            for idx, [module, module_name] in enumerate(modules):
                em = EditLinear(module, module_name, idx, aux_model, self.cfg, None)
                edit_modules.append(em)
                # autograd_params: 因为要获取的__delta__梯度是bias梯度的中间节点
                assert module.bias != None 
                autograd_params.append(module.bias)
        return edit_modules, aux_models, autograd_params

    def clear_module_deltas(self, x:bool, delta:bool, edit_weight:bool):
        for em in self.edit_modules:
            em.clear_delta(x, delta, edit_weight)
    
    def update_weights_and_clear_deltas(self):
        for em in self.edit_modules:
            em.update_weight_and_clear_delta()

    def restore_to_original_weights(self):
        for em in self.edit_modules:
            em.restore_to_original_weight()

    # def remove_module_hooks(self, x_handle:bool, delta_handle:bool, edit_handle:bool):
    #     for em in self.edit_modules:
    #         em.remove_hooks(x_handle, delta_handle, edit_handle)
            
    # def register_module_hooks(self):
    #     for em in self.edit_modules: 
    #         em.register_hooks()

    ############################################################################
    #                       以下为训练代码 (可复用)                             #
    ############################################################################
    def train_init(self, records_dir:str = 'train_records', train_name:str = None, 
            load_ckpt_path:str = None, save_ckpt_per_i = 1000, log_per_i = 10,
            discard_loss_multiple = 5, discard_loss_threshold = 10):
        # 初始化记录文件夹
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        train_name = train_name if train_name else t
        records_dir = os.path.join(records_dir, 'mend', self.cfg.edit_model_name, train_name)
        self.save_ckpt_dir = os.path.join(records_dir, 'checkpoints')
        if not os.path.exists(self.save_ckpt_dir):
            os.makedirs(self.save_ckpt_dir)
        logs_path = os.path.join(records_dir, 'logs')
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        with open(os.path.join(records_dir, 'config.yaml'), 'w') as f:
            yaml.dump(asdict(self.cfg), f)
        self.log_writer = SummaryWriter(logs_path)
        self.save_ckpt_per_i = save_ckpt_per_i
        self.log_per_i = log_per_i
        # 初始化优化器，读取预训练权重、已迭代次数
        self.discard_loss_multiple = discard_loss_multiple
        self.discard_loss_threshold = discard_loss_threshold
        self.aux_models_opt = Adam(self.aux_models.parameters(), self.cfg.aux_model.lr)
        self.edit_lrs_opt = Adam(self.edit_lrs.parameters(), self.cfg.edit_lr_lr)
        if load_ckpt_path and os.path.isfile(load_ckpt_path):
            self.train_i, self.train_epoch, self.last_loss = self.load_ckpt(load_ckpt_path, True)
        else:
            self.train_i, self.train_epoch, self.last_loss = 1, 1, 9999 

    def set_train(self, if_train = False):
        self.model.train(False)
        self.model.requires_grad_(False)
        self.autograd_params.requires_grad_(False)
        self.autograd_params.train(False)
        self.aux_models.requires_grad_(if_train)
        self.aux_models.train(if_train)
        self.edit_lrs.requires_grad_(if_train)
        self.edit_lrs.train(if_train)

    def train(self, epochs, data_generator:ParallelDataset):
        if self.log_writer == None:
            raise "Call `self.train_init()` to initialize training first!"
        self.set_train(True)
        start_epoch = self.train_epoch
        for self.train_epoch in range(start_epoch, epochs + 1):
            for d in tqdm(data_generator):
                loss, relia_loss, gen_losses, loc_losses = self.__train_a_batch__(*d)
                if loss != None: 
                    if self.train_i % self.log_per_i == 0:
                        self.write_logs(self.train_i, {
                            'Epoch': self.train_epoch,
                            'Loss': loss,
                            'Reliability loss': relia_loss,
                            'Generality loss': gen_losses,
                            'Locality loss': loc_losses
                        })
                    if self.train_i % self.save_ckpt_per_i == 0:
                        self.save_ckpt(self.train_i, self.train_epoch, loss)
                self.train_i += 1 
        self.set_train(False)
                    
    def __train_a_batch__(self, edit_xym:Tuple, gen_xym:Dict[str, Tuple], 
                      loc_xm:Dict[str, Tuple]):
        '''Assume:
        edit_xym: (input_ids, label_ids, masks)
        gen_xym: {
            loss_name_1: (input_ids, label_ids, masks),
            loss_name_2: (input_ids, label_ids, masks), ...
        }
        loc_xm: {
            loss_name_1: (input_ids, masks)
            loss_name_2: (input_ids, masks), ...
        }
        ''' 
        self.clear_module_deltas(True, True, True)
        # prediction before edit for locality loss
        with torch.no_grad():
            for loss_name, sp in loc_xm.items():
                input_ids, masks = sp
                pre_logits = self.model(input_ids).logits
                loc_xm[loss_name] = (sp, pre_logits)
        # edit
        (input_ids, label_ids, masks) = edit_xym
        self.__edit_batch__(input_ids, label_ids, masks)
        # compute reliability loss
        relia_loss = self.cfg.relia_lambda * label_loss(self.model, input_ids, label_ids, masks)
        # init losses
        loss = 0
        loss += relia_loss
        # compute generality loss
        gen_losses = {}
        for loss_name, sp in gen_xym.items():
            input_ids, label_ids, masks = sp
            gen_loss = self.cfg.gen_lambda * label_loss(self.model, input_ids, label_ids, masks)
            gen_losses[loss_name] = gen_loss
            loss += gen_loss 
        # compute locality loss
        loc_losses = {}
        for loss_name, sp in loc_xm.items():
            (input_ids, masks), pre_logits = sp
            post_logits = self.model(input_ids).logits
            loc_loss = self.cfg.loc_lambda * logit_KL_loss(pre_logits, post_logits, masks)
            loc_losses[loss_name] = loc_loss
            loss += loc_loss
        # discard abnormal loss
        if loss > self.discard_loss_threshold and \
            loss > self.discard_loss_multiple * self.last_loss:
            print('One training batch was discarded at %d iterations. Loss = %.2f.'%\
                  (self.train_i, loss))
            return None, None, None
        loss.backward()
        # try gradient clip
        torch.nn.utils.clip_grad_norm_(self.aux_models.parameters(), 100.,
                                                  error_if_nonfinite=True)
        self.aux_models_opt.step()
        self.edit_lrs_opt.step()
        self.aux_models_opt.zero_grad()
        self.edit_lrs_opt.zero_grad()
        self.last_loss = loss.detach()
        return loss, relia_loss, gen_losses, loc_losses

    def write_logs(self, i, logs:dict):
        for log_name, log in logs.items():
            if type(log) == dict:
                logs1 = {}
                for n, l in log.items():
                    logs1[log_name + '-' + n] = l
                self.write_logs(i, logs1)
            else:   
                self.log_writer.add_scalar(log_name, log, i)


    def save_ckpt(self, i:int, epoch:int, loss:float):
        ckpt_name = 'epoch-%d-i-%d-loss-%.4f'%(epoch, i, loss)
        ckpt_path = os.path.join(self.save_ckpt_dir, ckpt_name)
        ckpt = {
            'i': i,
            'epoch': epoch,
            'loss': loss,
            'aux_models': self.aux_models.state_dict(),
            'edit_lrs':self.edit_lrs.state_dict(),
            'aux_models_opt': self.aux_models_opt.state_dict(),
            'edit_lrs_opt': self.edit_lrs_opt.state_dict(),
        }
        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, ckpt_path, load_opt = False):
        ckpt = torch.load(ckpt_path)
        self.aux_models.load_state_dict(ckpt['aux_models'])
        self.edit_lrs.load_state_dict(ckpt['edit_lrs'])
        if load_opt:
            self.aux_models_opt.load_state_dict(ckpt['aux_models_opt'])
            self.edit_lrs_opt.load_state_dict(ckpt['edit_lrs_opt'])
        print('Load mend checkpoints from', ckpt_path)
        return ckpt['i'], ckpt['epoch'], ckpt['loss']



def find_module(module:nn.Module, module_path:List[str]):
    for comp in module_path:
        if hasattr(module, comp):
            module = getattr(module, comp)
        elif comp.isdigit():
            module = module[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    return module
 
def label_loss(model, input_ids:torch.Tensor, label_ids:torch.Tensor, masks:torch.Tensor):
    # input_ids/label_ids/masks: [batch, max_len]
    logits = model(input_ids).logits
    log_pre_p = torch.log_softmax(logits, 2) # [batch, max_len, voc_size]
    log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, max_len]
    loss = -(log_pre_p * masks).sum()/masks.sum() 
    return loss

def logit_KL_loss(logits1:torch.Tensor, logits2:torch.Tensor, masks:torch.Tensor):
    # logits1/logits2: [batch, max_len, voc_size], masks: [batch, max_len]
    log_p1 = torch.log_softmax(logits1, 2)
    log_p2 = torch.log_softmax(logits2, 2)
    p1 = torch.softmax(logits1, 2)
    kl_loss = (p1 * (log_p1 - log_p2)).sum(2) # [batch, max_len]
    loss = (kl_loss * masks).sum()/masks.sum() 
    return loss

