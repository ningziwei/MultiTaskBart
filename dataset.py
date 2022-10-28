import torch
import bisect
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler

class CoNLLDataset(Dataset):
    '''输入数据的迭代器'''
    def __init__(self, sentences, data_dealer):
        super(CoNLLDataset, self).__init__()
        data = []
        for sent in sentences:
            data.append(data_dealer.get_one_sample(sent))
        self.data = data
    
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

class GroupBatchRandomSampler(Sampler):
    '''
    得到按长度分组后采样的样本序号
    '''
    def __init__(self, data_source, batch_size, group_interval):
        '''
        data_source: 可迭代对象
        batch_size: batch的大小
        group_interval: 为了减少pad，把样本按长度分组，在同一组中执行BatchSampler
        '''
        super(GroupBatchRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.group_interval = group_interval

        max_len = max([len(d['head']["enc_src_ids"]) for d in self.data_source])
        breakpoints = np.arange(group_interval, max_len, group_interval)
        self.groups = [[] for _ in range(len(breakpoints) + 1)]
        for i, data in enumerate(self.data_source):
            group_id = bisect.bisect_right(breakpoints, len(data['head']["enc_src_ids"]))
            self.groups[group_id].append(i)
        self.batch_indices = []
        for g in self.groups:
            self.batch_indices.extend(list(
                BatchSampler(SubsetRandomSampler(g), 
                self.batch_size, False)
            ))
    
    def __iter__(self):
        batch_indices = []
        for g in self.groups:
            batch_indices.extend(list(
                BatchSampler(SubsetRandomSampler(g), self.batch_size, False)
            ))
        return (batch_indices[i] for i in torch.randperm(len(batch_indices)))
    
    def __len__(self):
        return len(self.batch_indices)

def collate_fn(batch_data, config):
    '''根据batch结果实时生成模型的输入数据，避免内存压力太大'''
    pad_value = config['pad_value']
    device = config['device']
    padded_batch = defaultdict(list)

    batch_data = [b['head'] for b in batch_data]
    if config['src_self_sup']:
        txt_ids, mask = padding(
            [d["txt_ids"] for d in batch_data], pad_value)
        padded_batch["txt_ids"] = torch.tensor(txt_ids, dtype=torch.long, device=device)
        txt_len = [d["txt_len"] for d in batch_data]
        padded_batch["txt_len"] = torch.tensor(txt_len, dtype=torch.long, device=device)
        padded_batch["txt_mask"] = torch.tensor(mask, dtype=torch.bool, device=device)

    enc_src_ids, mask = padding(
        [d["enc_src_ids"] for d in batch_data], pad_value)
    padded_batch["enc_src_ids"] = torch.tensor(enc_src_ids, dtype=torch.long, device=device)
    enc_src_len = [d["enc_src_len"] for d in batch_data]
    padded_batch["enc_src_len"] = torch.tensor(enc_src_len, dtype=torch.long, device=device)
    padded_batch["enc_mask"] = torch.tensor(mask, dtype=torch.bool, device=device)
    if config['enc_attn_mask']:
        enc_attn_mask = get_enc_attn_mask(batch_data[0]['cls_toks_num'], torch.tensor(mask))
        padded_batch["enc_attn_mask"] = torch.tensor(enc_attn_mask, dtype=torch.bool, device=device)
    else:
        padded_batch["enc_attn_mask"] = None
    
    for i in range(len(batch_data[0]['dec_src_ids'])):
        dec_src_ids, mask = padding(
            [d["dec_src_ids"][i] for d in batch_data], pad_value)
        dec_src_pos, mask = padding(
            [d["dec_src_pos"][i] for d in batch_data], pad_value)
        padded_batch["dec_src_ids_bund"].append(torch.tensor(dec_src_ids, dtype=torch.long, device=device))
        padded_batch["dec_src_pos_bund"].append(torch.tensor(dec_src_pos, dtype=torch.long, device=device))
        padded_batch["dec_mask_bund"].append(torch.tensor(mask, dtype=torch.bool, device=device))
        
        dec_targ_pos, mask = padding(
            [d["dec_targ_pos"][i] for d in batch_data], pad_value)
        padded_batch["dec_targ_pos_bund"].append(torch.tensor(dec_targ_pos, dtype=torch.long, device=device))

    targ_ents = [d["targ_ents"] for d in batch_data]
    padded_batch["targ_ents"] = targ_ents
    return padded_batch

def padding(data, pad_value=0, dim=2):
    '''
    pad data to maximum length
    data: list(list), unpadded data
    pad_value: int, filled value
    dim: int, dimension of padded data
        dim=2, result=(batch_size, sen_len)
        dim=3, result=(batch_size, sen_len, word_len)
    '''
    sen_len = max([len(d) for d in data])
    if dim == 2:
        padded_data = [d + [pad_value] * (sen_len-len(d)) for d in data]
        padded_mask = [[1] * len(d) + [0] * (sen_len-len(d)) for d in data]
        return padded_data, padded_mask
    elif dim == 3:
        word_len = max([max([len(dd) for dd in d]) for d in data])
        padded_data = []
        padded_mask = []
        for d in data:
            padded_data.append([])
            padded_mask.append([])
            for dd in d:
                padded_data[-1].append(dd + [pad_value] * (word_len-len(dd)))
                padded_mask[-1].append([1] * len(dd) + [0] * (word_len-len(dd)))
            for _ in range(sen_len - len(d)):
                padded_data[-1].append([pad_value] * word_len)
                padded_mask[-1].append([0] * word_len)
        return padded_data, padded_mask
    else:
        raise NotImplementedError("Dimension %d not supported! Legal option: 2 or 3." % dim)

def get_enc_attn_mask(toks_num, enc_padding_mask):
    '''
    生成encoder中要用到的注意力掩码矩阵
    toks_num: int
    enc_padding_mask: bsz*max_len
    attn_mask: bsz*max_len*max_len
    '''
    mask = enc_padding_mask.unsqueeze(-1)
    attn_mask = torch.matmul(mask, mask.permute(0,2,1))
    attn_mask[:,:,0] = 1
    attn_mask[:,1:1+toks_num,1:1+toks_num] = 0
    attn_mask[:,range(1,1+toks_num),range(1,1+toks_num)] = 1
    return attn_mask

def flat_sequence(
    batch_pred,
    batch_enc_src_ids,
    batch_dec_src_ids,
    batch_dec_src_pos,
    dic_pos_cls,
    pad_value=0,
    device=torch.device("cpu")
):
    '''
    根据解码结果，将序列拉平
    '''
    next_ids = []
    next_pos = []
    for i in range(len(batch_pred)):
        pred = batch_pred[i]
        enc_src_ids = batch_enc_src_ids[i]
        dec_src_ids = batch_dec_src_ids[i]
        dec_src_pos = batch_dec_src_pos[i]
        next_dec_src_ids = []
        next_dec_src_pos = []
        for j in range(len(dec_src_ids)):
            if dec_src_ids[j]==-1: continue
            # print(pred[j], enc_src_ids[pred[j]-1])
            next_dec_src_ids.append(dec_src_ids[j].item())
            next_dec_src_pos.append(dec_src_pos[j].item())
            if pred[j] in dic_pos_cls:
                next_dec_src_ids.append(enc_src_ids[pred[j]-1].item())
                next_dec_src_pos.append(pred[j].item())
        next_ids.append(next_dec_src_ids)
        next_pos.append(next_dec_src_pos)
    next_ids_padded, mask = padding(next_ids, pad_value)
    next_ids_padded = torch.tensor(next_ids_padded, dtype=torch.long, device=device)
    next_dec_mask = torch.tensor(mask, dtype=torch.bool, device=device)
    next_pos_padded, _ = padding(next_pos, pad_value)
    next_pos_padded = torch.tensor(next_pos_padded, dtype=torch.long, device=device)
    return next_ids_padded, next_pos_padded, next_dec_mask, next_pos
