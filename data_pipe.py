
import json
import torch
import opencc
# from transformers import BertTokenizer
from model.tokenizer_full import BertTokenizer

def parse_CoNLL_file(filename):
    '''
    加载CoNLL格式的数据集
    sentences: [
        [
            {'word':'感','tag':'o'},{'word':'动','tag':'o'},
            {'word':'中','tag':'b-loc.nam'},{'word':'国','tag':'i-loc.nam'}
        ]
    ]
    '''
    sentences = [] 
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()
    for line in lines:
        line = line.strip()
        # print(repr(line))
        if not line:
            if not len(sentences):
                '''去掉开头的空行'''
                continue
            if len(sentences[-1]):
                '''句子后的第一个空行'''
                sentences.append([])
                continue
            if not len(sentences[-1]):
                '''空行前还是空行'''
                continue
        if line and not len(sentences):
            '''开头第一个token'''
            sentences.append([])
            continue
        line_strlist = line.split()
        if line_strlist[0] != "-DOCSTART-":
            word = line_strlist[0]
            tag = line_strlist[-1].lower()
            sentences[-1].append({'word':word.lower(), 'tag':tag})
    sentences = [s for s in sentences if len(s)]
    return sentences

def parse_label(sentences, config, cls_token_path=None):
    '''
    得到实体抽取数据集的所有标签和实体类别
    new_tokens_bundle: [
        ['<<loc.nam-s>>', '<<loc.nam-e>>', '<<ent_end>>'],
        ['<<loc.nam-s>>'],
        ['<<loc.nam-e>>', '<<ent_end>>']
    ]
    '''
    label_dic = {}
    for sent in sentences:
        for s in sent:
            label_dic[s['tag']] = True
    classes = [
        lab[2:] for lab in label_dic if '-' in lab]
    if config['cls_type']=='cls_e_cls':
        cls_tok_dic = {
            lab: [f'<<{lab}-s>>', f'<<{lab}-e>>'] for lab in classes
        }
    elif config['cls_type']=='s_e_cls':
        cls_tok_dic = {
            lab: [f'<<lab-s>>', f'<<{lab}-e>>'] for lab in classes
        }
    elif config['cls_type']=='s_e':
        cls_tok_dic = {
            lab: [f'<<lab-s>>', f'<<lab-e>>'] for lab in classes
        }
    
    new_tokens = []
    start_tokens = []
    end_tokens = []
    ent_end_tok='<<ent_end>>'
    ent_end_token = []
    for _, v in cls_tok_dic.items():
        new_tokens += v
        start_tokens.append(v[0])
        end_tokens.append(v[1])
    new_tokens = list(set(new_tokens))
    start_tokens = list(set(start_tokens))
    end_tokens = list(set(end_tokens))
    if config['fold']==3:
        new_tokens.append(ent_end_tok)
        ent_end_token.append(ent_end_tok)
    cls_token_cache = {
        'cls_tok_dic': cls_tok_dic,
        'new_tokens_bundle': [new_tokens, start_tokens, end_tokens, ent_end_token]
    }
    if cls_token_path is not None:
        with open(cls_token_path, 'w', encoding='utf8') as f:
            json.dump(cls_token_cache, f)
    return cls_token_cache

def parse_txt(tokenizer, sent):
    '''处理预测过程中没有标签的数据'''
    tokens = tokenizer.tokenize(sent)
    return [{'word':tok,'tag':'o'} for tok in tokens]

class MyTokenizer(BertTokenizer):
    def add_special_tokens(self, cls_tok_dic, new_tokens_bundle):
        '''将表示实体边界的特殊标记添加到分词器中''' 
        # for x in new_tokens_bundle: print('109', x)
        self.cls_tok_dic = cls_tok_dic
        new_tokens, start_tokens, end_tokens, ent_end_token = new_tokens_bundle
        self.unique_no_split_tokens += new_tokens
        self.add_tokens(new_tokens)

        dic_tok_id = {}
        dic_tok_order = {}
        num = 0
        for tok in new_tokens:
            dic_tok_id[tok] = self.convert_tokens_to_ids(tok)
            # dic_tok_order[tok] = len(dic_tok_order)
            if '-s' in tok:
                dic_tok_order[tok] = num
                end_tok = tok.replace('-s', '-e')
                dic_tok_order[end_tok] = num + len(start_tokens)
                num += 1
        dic_tok_pos = {k:v+2 for k,v in dic_tok_order.items()}
        # print('dic_tok_pos', dic_tok_pos)
        dic_start_pos_tok = {dic_tok_pos[k]:k for k in start_tokens}
        dic_end_pos_tok = {dic_tok_pos[k]:k for k in end_tokens}
        dic_ent_end_pos_tok = {dic_tok_pos[k]:k for k in ent_end_token}
        dic_all_end_pos_tok = dic_end_pos_tok.copy()
        if len(ent_end_token):
            dic_all_end_pos_tok[dic_tok_pos[ent_end_token[0]]] = ent_end_token[0]

        self.dic_tok_id = dic_tok_id
        self.dic_tok_order = dic_tok_order
        self.dic_tok_pos = dic_tok_pos
        self.dic_hir_pos_cls = [dic_start_pos_tok, dic_end_pos_tok]
        self.dic_start_pos_tok = dic_start_pos_tok
        self.dic_ent_end_pos_tok = dic_ent_end_pos_tok
        self.dic_end_pos_tok = dic_end_pos_tok
        self.dic_all_end_pos_tok = dic_all_end_pos_tok
        
class DataDealer:
    def __init__(self, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        rotate_pos_cls = []
        rotate_pos_cls.append(tokenizer.dic_start_pos_tok)
        rotate_pos_cls.append(tokenizer.dic_all_end_pos_tok)
        self.rotate_pos_cls = rotate_pos_cls
        self.tokens_to_ids = self.tokenizer.convert_tokens_to_ids


    def get_one_sample(self, sent):
        '''
        编码器和解码器输入都带有prompt，用pointer的方式找到实体的开始位置和结束位置
        生成一个样本的输入数据和目标数据
        sent_bund: [ 解码一步后的实际输出
            ['[CLS]', '感', '动', '中', '国', '[SEP]'], 
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '[SEP]'], 
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', '[SEP]'], 
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', '<<loc.nam-e>>', '[SEP]']]
        targ_bund: [ decoder输出序列对应的中间结果
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '国', '[SEP]'], 
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', '[SEP]'], 
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', '<<loc.nam-e>>', '[SEP]']]
        sent_ids_bund: [ sent_bund中token转id
            [101, 2697, 1220, 704, 1744, 102], 
            [101, 2697, 1220, 21136, 704, 1744, 102], 
            [101, 2697, 1220, 21136, 704, 1744, 21144, 102], 
            [101, 2697, 1220, 21136, 704, 1744, 21144, 21137, 102]]
        sent_pos_bund: [ sent_bund中token转pos
            [0, 19, 20, 21, 22, 1], 
            [0, 19, 20, 10, 21, 22, 1], 
            [0, 19, 20, 10, 21, 22, 18, 1], 
            [0, 19, 20, 10, 21, 22, 18, 11, 1]]
        targ_ids_bund: [ targ_bund中token转id
            [101, 2697, 1220, 21136, 1744, 102], 
            [101, 2697, 1220, 21136, 704, 1744, 21144, 102], 
            [101, 2697, 1220, 21136, 704, 1744, 21144, 21137, 102]] 
        targ_pos_bund: [ targ_bund中token转pos
            [0, 19, 20, 10, 22, 1], 
            [0, 19, 20, 10, 21, 22, 18, 1], 
            [0, 19, 20, 10, 21, 22, 18, 11, 1]]
        
        return: {
            'raw_chars': ['[CLS]', '感', '动', '中', '国', '[SEP]'], 
            'src_toks': [
                '[CLS]', '<<per.nam-s>>', '<<per.nam-e>>', '<<gpe.nam-s>>', 
                '<<gpe.nam-e>>', '<<org.nom-s>>', '<<org.nom-e>>', '<<per.nom-s>>', 
                '<<per.nom-e>>', '<<loc.nam-s>>', '<<loc.nam-e>>', '<<loc.nom-s>>', 
                '<<loc.nom-e>>', '<<gpe.nom-s>>', '<<gpe.nom-e>>', '<<org.nam-s>>', 
                '<<org.nam-e>>', '<<ent_end>>', '感', '动', '中', '国', '[SEP]'], 
            'targ_toks': [
                '[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', 
                '<<loc.nam-e>>', '[SEP]'], 
            'enc_src_ids': [
                101, 21128, 21129, 21130, 21131, 21132, 21133, 21134, 21135, 21136, 
                21137, 21138, 21139, 21140, 21141, 21142, 21143, 21144, 2697, 1220, 
                704, 1744, 102], 
            'enc_src_len': 23, 
            'dec_src_ids': [
                [101, 2697, 1220, 704, 1744], 
                [101, 2697, 1220, 21136, 704, 1744, 102], 
                [101, 2697, 1220, 21136, 704, 1744, 21144, 102]], 
            'dec_targ_pos': [
                [19, 20, 10, 22, 1], 
                [19, 20, 10, 21, 22, 18, 1], 
                [19, 20, 10, 21, 22, 18, 11, 1]], 
            'targ_ents': [[10, 21, 22, 18, 11]]
        }
        '''
        # print(sent)
        word_shift = len(self.tokenizer.dic_tok_pos) + 2
        for i, s in enumerate(sent): 
            s['pos'] = i + word_shift
        # print('210', word_shift, sent)
        head_sent, head_targ, tail_sent, tail_targ = self.get_semi_sent(sent)
        head_dic_sent_bund = [sent, head_sent]
        head_dic_targ_bund = [head_targ]
        head_task = self.get_one_task(head_dic_sent_bund, head_dic_targ_bund, 'head')

        tail_dic_sent_bund = [sent, tail_sent]
        tail_dic_targ_bund = [tail_targ]
        # print('tail_sent', tail_sent)
        # print('tail_targ', tail_targ)
        tail_task = self.get_one_task(tail_dic_sent_bund, tail_dic_targ_bund, 'end')

        # print('head_task', head_task)
        # print('tail_task', tail_task)
        ent_seq = get_ents_seq_from_semi_seq(
            head_seq=head_task['dec_targ_pos'][0], 
            tail_seq=tail_task['dec_targ_pos'][0], 
            dec_src=head_task['dec_src_pos'][0], 
            rotate_pos_cls=self.rotate_pos_cls
        )
        targ_ents = get_targ_ents_2(
            ent_seq, self.rotate_pos_cls
        )
        class_task = {
            'ent_seq': ent_seq, 
            'targ_ents': targ_ents
        }

        sample = {
            'head': head_task,
            'tail': tail_task,
            'cls': class_task
        }
        # print('221', head_task)
        # print('222', tail_task)
        return sample

    def get_semi_sent(self, sent):
        '''
        得到有一半边界的句子对应的输入输出序列
        认为0对应bos，1对应eos，特殊标记从2开始算，特殊标记后面才是原始文本
        输入: 源文本sent
        sent: [
            {'word':'感','tag':'O'},{'word':'动','tag':'O'},
            {'word':'中','tag':'B-LOC'},{'word':'国','tag':'I-LOC'}]
        输出: 渐进式序列生成任务解码器的输入和输出序列及对应位置序列
        sent_bund: [
            ['感', '动', '中', '国'], 
            ['感', '动', '[loc.nam-s]', '中', '国'],
            ['感', '动', '[loc.nam-s]', '中', '国', '[loc.nam-e]']]
        targ_bund: [
            ['感', '动', '<<loc.nam-s>>', '国'], 
            ['感', '动', '<<loc.nam-s>>', '中', '国', '<<loc.nam-e>>']]
        sent_pos_bund: [
            [19, 20, 21, 22], 
            [19, 20, 10, 21, 22], 
            [19, 20, 10, 21, 22, 11]]
        targ_pos_bund:  [
            [19, 20, 10, 22], 
            [19, 20, 10, 21, 22, 11]]
        '''
        last_w = {'word':'', 'tag':'o'}
        cls_tok_dic = self.tokenizer.cls_tok_dic
        dic_tok_pos = self.tokenizer.dic_tok_pos

        sent = sent + [last_w]
        targ_sent = sent[1:] + [last_w]

        head_sent = []
        head_targ = []
        # 添加实体开始标记
        for w, w_tar in zip(sent, targ_sent):
            if w['tag'].startswith('b-'):
                word = cls_tok_dic[w['tag'][2:]][0]
                head_sent.append({
                    'word':word,'tag':'start','pos':dic_tok_pos[word]
                })
                head_targ = head_targ[:-1]
                head_targ.append(head_sent[-1])
            if w_tar['word']:
                head_targ.append(w_tar)
            if w['word']:
                head_sent.append(w)
        
        tail_sent = []
        tail_targ = []
        w_ = {'word':'', 'tag':''}
        # 添加实体结束标记
        for w, w_tar in zip(sent, targ_sent):
            if w['tag'][:2] in ['o','b-'] and w_['tag'][:2] in ['i-','b-']:  
                word = cls_tok_dic[w_['tag'][2:]][1]
                tail_sent.append({
                    'word':word,'tag':'end','pos':dic_tok_pos[word]
                })
                if w['word']: tail_targ = tail_targ[:-1]
                tail_targ.append(tail_sent[-1])
            if w_tar['word']:
                tail_targ.append(w_tar)
            if w['word']:
                tail_sent.append(w)
            w_ = w
        
        def _add_head(sent, targ):
            '''
            在目标序列中添加第一个字，保证序列的完整性
            开始标记在句首时，此时targ为空，所以targ[:-1]不会变短
            其他情况下，每遇见一个特殊标记，targ就比sent短一个字符
            由于错位生成的原因，targ的长度本来就短一个
            所以，开始标记在targ首时，不用给targ前面补[CLS]要生成的字符
            其他情况下都需要补[CLS]要生成的字符
            '''
            if not len(targ) or targ[0]['tag'] not in dic_tok_pos:
                targ = [sent[0]] + targ
            return targ
        
        head_targ = _add_head(head_sent, head_targ)
        tail_targ = _add_head(tail_sent, tail_targ)

        return head_sent, head_targ, tail_sent, tail_targ

    def get_one_task(self, dic_sent_bund, dic_targ_bund, task_type):
        '''得到一个子任务的全部输入输出数据'''
        sent_bund, sent_pos_bund = [], []
        targ_bund, targ_pos_bund = [], []
        for i in range(len(dic_sent_bund)):
            sent = dic_sent_bund[i]
            sent_bund.append([s['word'] for s in sent])
            sent_pos_bund.append([s['pos'] for s in sent])
        for i in range(len(dic_targ_bund)):
            targ = dic_targ_bund[i]
            targ_bund.append([s['word'] for s in targ])
            targ_pos_bund.append([s['pos'] for s in targ])
        # print('338', sent_bund, sent_pos_bund)
        # print('339', targ_bund, targ_pos_bund)
        if self.config['targ_self_sup']:
            '''对完整的目标序列执行一次自编码'''
            targ_bund.append(sent_bund[-1])
            targ_pos_bund.append(sent_pos_bund[-1])
        
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        sent_bund = [[bos_token]+sent+[eos_token] for sent in sent_bund]
        targ_bund = [[bos_token]+targ+[eos_token] for targ in targ_bund]
        sent_ids_bund = [self.tokens_to_ids(tks) for tks in sent_bund]
        targ_ids_bund = [self.tokens_to_ids(tks) for tks in targ_bund]
        sent_pos_bund = [[0]+pos+[1] for pos in sent_pos_bund]
        targ_pos_bund = [[0]+pos+[1] for pos in targ_pos_bund]

        dec_src_ids, dec_src_pos, dec_targ_pos = self.get_dec_src_tar(
            sent_bund, targ_bund,
            sent_ids_bund, sent_pos_bund, 
            targ_ids_bund, targ_pos_bund)
        
        '''在编码器的输入序列中添加表示实体的特殊标记'''
        pad_token = self.tokenizer.pad_token
        if task_type=='head':
            pos_cls = self.tokenizer.dic_start_pos_tok
            padded_prompt_pos = list(pos_cls.values()) + [pad_token]*len(pos_cls)
        else:
            pos_cls = self.tokenizer.dic_end_pos_tok
            # print('359', pos_cls)
            padded_prompt_pos = [pad_token]*len(pos_cls) + list(pos_cls.values())
        src_toks = sent_bund[0]
        txt_ids = [self.tokens_to_ids(tk) for tk in src_toks]
        src_toks = [src_toks[0]] + padded_prompt_pos + src_toks[1:]
        enc_src_ids = [self.tokens_to_ids(tk) for tk in src_toks]
        '''获得监督标签'''
        targ_ents = get_semi_ents(sent_pos_bund[-1], pos_cls, task_type)
        # print(list(pos_cls.keys()))
        return {
            'prompt_pos_list': list(pos_cls.keys()),
            'raw_chars': sent_bund[0],
            'src_toks': src_toks,
            'targ_toks': sent_bund[-1],
            'txt_ids': txt_ids,
            'txt_len': len(txt_ids),
            'enc_src_ids': enc_src_ids,
            'enc_src_len': len(enc_src_ids),
            'dec_src_ids': dec_src_ids,
            'dec_src_pos': dec_src_pos,
            'dec_targ_pos': dec_targ_pos,
            'targ_ents': targ_ents
        }
        
    def get_dec_src_tar(
        self, sent_bund, targ_bund,
        sent_ids_bund, sent_pos_bund, 
        targ_ids_bund, targ_pos_bund):
        '''
        得到解码器输入和目标的token、id、pos列表
        实际上有用的只有sent的ids和targ的pos
        所有序列都是以[CLS]开头，[SEP]结尾的，所以targ从1开始截取到最后，
        src根据从0开始截取targ的长度个
        '''
        dec_src_toks = []
        dec_targ_toks = []
        dec_src_ids = []
        dec_src_pos = []
        dec_targ_ids = []
        dec_targ_pos = []
        # print('278')
        for i in range(len(targ_pos_bund)):
            sent_toks = sent_bund[i]
            sent_ids = sent_ids_bund[i]
            sent_pos = sent_pos_bund[i]
            targ_toks = targ_bund[i]
            targ_ids = targ_ids_bund[i]
            targ_pos = targ_pos_bund[i]
            # 目标序列的最后一位一定要是[SEP]
            dec_src_toks.append(sent_toks[:len(targ_pos)-1])
            dec_src_ids.append(sent_ids[:len(targ_pos)-1])
            dec_src_pos.append(sent_pos[:len(targ_pos)-1])
            dec_targ_toks.append(targ_toks[1:])
            dec_targ_ids.append(targ_ids[1:])
            dec_targ_pos.append(targ_pos[1:])
            # print(dec_src_toks[-1])
            # print(dec_targ_toks[-1])
            # print(dec_src_ids[-1], dec_targ_ids[-1])
            
        return dec_src_ids, dec_src_pos, dec_targ_pos


def get_semi_ents(pos_list, pos_cls, task_type):
    '''获得实体一半的边界结果'''
    ents = []
    for i,pos in enumerate(pos_list):
        if pos==-1: break
        if pos in pos_cls:
            if task_type=='head':
                ents.append(pos_list[i:i+2])
            else:
                ents.append(pos_list[i-1:i+1])
    return ents

def get_ents_seq_from_semi_seq(
    head_seq, tail_seq, dec_src, rotate_pos_cls
):
    '''根据两个半边界序列得到完整的序列'''
    print('data_pipe 451', len(head_seq), head_seq)
    print(len(tail_seq), tail_seq)
    print(len(dec_src), dec_src)
    ent_seq = []
    for i in range(len(dec_src)):
        if dec_src[i]==-1: break
        ent_seq.append(dec_src[i])
        if tail_seq[i] in rotate_pos_cls[1]:
            ent_seq.append(tail_seq[i])
        if head_seq[i] in rotate_pos_cls[0]:
            ent_seq.append(head_seq[i])
    return ent_seq

def get_targ_ents(pos_list, rotate_pos_cls):
    '''得到序列中的实体'''
    i, N = 0, len(pos_list)

    ents = []
    while i<N:
        # 碰到实体开始符
        if pos_list[i] in rotate_pos_cls[0]:
            ent = [pos_list[i]]
            i += 1
            while i<N:
                ent.append(pos_list[i])
                # 碰到实体结束符
                if pos_list[i] in rotate_pos_cls[1]:
                    i += 1
                    while i<N:
                        if pos_list[i] in rotate_pos_cls[1]:
                            ent.append(pos_list[i])
                        else:
                            break
                        i += 1
                    break
                i += 1
            ents.append(ent)
        else:
            i += 1
    # print('431', ents)
    return ents

def get_targ_ents_3(pos_list, rotate_pos_cls, ent_end_pos):
    '''得到序列中的实体'''
    i, N = 0, len(pos_list)

    ents = []
    while i<N:
        # 碰到实体开始符
        if pos_list[i] in rotate_pos_cls[0]:
            ent = [pos_list[i]]
            i += 1
            while i<N:
                ent.append(pos_list[i])
                # 碰到实体结束符
                if pos_list[i] == ent_end_pos:
                    i += 1
                    if i<N and pos_list[i] in rotate_pos_cls[1]:
                        ent.append(pos_list[i])
                        i += 1
                    break
                i += 1
            ents.append(ent)
        else:
            i += 1
    return ents

def get_targ_ents_2(pos_list, rotate_pos_cls, ent_end_pos=None):
    '''得到序列中的实体'''
    i, N = 0, len(pos_list)

    ents = []
    while i<N:
        # 碰到实体开始符
        if pos_list[i] in rotate_pos_cls[0]:
            ent = [pos_list[i]]
            i += 1
            while i<N:
                ent.append(pos_list[i])
                # 碰到实体结束符
                if pos_list[i] in rotate_pos_cls[1]:
                    i += 1
                    break
                i += 1
            ents.append(ent)
        else:
            i += 1
    return ents

def merge_bound_token(src_seq, start_seq, end_seq):
    '''
    合并只有一个边界的序列
    先合并结束标记，再合并开始标记
    '''
    merge_seq = []
    for i in range(len(src_seq)):
        merge_seq.append(src_seq[i])
        if end_seq[i]['tag']=='end':
            merge_seq.append(end_seq[i])
        if start_seq[i]['tag']=='start':
            merge_seq.append(start_seq[i])
    return merge_seq


    









