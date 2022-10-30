import os
import json
import time
import torch
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import PretrainedConfig
from transformers import get_linear_schedule_with_warmup

import utils
from data_pipe import *
from dataset import *
# from transformers import BartModel
from model.hi_bart import HiBart
from model.losses import CrossEntropyLossWithMask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def get_logger_dir(config):
    '''
    input
        config: 配置参数
    output
        logger: 日志记录仪
        OUTPUT_DIR: 模型存储路径
    '''
    output_path = config["output_path"]
    prefix = config['dataset'].split('_')[0]+'_'
    curr_time = time.strftime("%m%d%H%M", time.localtime())
    OUTPUT_DIR = os.path.join(output_path, prefix + curr_time)
    if os.path.exists(OUTPUT_DIR):
        os.system("rm -rf %s" % OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    os.system("cp %s %s" % (config['config_path'], OUTPUT_DIR))
    logger = utils.Logger(open(os.path.join(OUTPUT_DIR, "log.txt"), 'w'))
    length = max([len(arg) for arg in config.keys()])
    for arg, value in config.items():
        logger("%s | %s" % (arg.ljust(length).replace('_', ' '), str(value)))
    return logger, OUTPUT_DIR

def get_tokenizer(config):
    '''
    解析训练集标签，得到模型添加特殊标记的tokenizer
    '''
    dataset_dir = config['dataset_dir']
    cls_token_path = os.path.join(
        dataset_dir, '{}-{}.json'.format(config['cls_type'], config['fold']))
    if not os.path.exists(cls_token_path):
        file_path = os.path.join(dataset_dir, 'train.train')
        sentences = parse_CoNLL_file(file_path)
        cls_token_cache = parse_label(sentences, config, cls_token_path)
    else:
        with open(cls_token_path, encoding="utf-8") as fp:
            cls_token_cache = json.load(fp)
    cls_tok_dic = cls_token_cache['cls_tok_dic']
    new_tokens_bundle = cls_token_cache['new_tokens_bundle']
    my_tokenizer = MyTokenizer.from_pretrained(config['model_path'])
    my_tokenizer.add_special_tokens(cls_tok_dic, new_tokens_bundle)
    
    return my_tokenizer

def get_data_loader(config, data_dealer):
    '''得到三个数据集的dataloader'''
    def get_loader(subset):
        file_path = os.path.join(config['dataset_dir'], f'{subset}.{subset}')
        sentences = parse_CoNLL_file(file_path)
        __dataset = CoNLLDataset(sentences, data_dealer)
        __sampler = GroupBatchRandomSampler(
            __dataset, config["batch_size"], config["group_interval"])
        __loader = DataLoader(
            dataset=__dataset, 
            batch_sampler=__sampler, 
            collate_fn=lambda x: collate_fn(x, config))
        return __loader
    
    train_loader = get_loader("train")
    test_loader = get_loader("test")
    valid_loader = get_loader("dev")
    return train_loader, test_loader, valid_loader

def init_cls_token_triv(bart, dic_tok_id, triv_tokenizer):
    '''在bart模型中初始化特殊标记的嵌入向量'''
    num_tokens, _ = bart.encoder.embed_tokens.weight.shape
    bart.resize_token_embeddings(len(dic_tok_id)+num_tokens)
    for tok, val in dic_tok_id.items():
        char_idx = triv_tokenizer.convert_tokens_to_ids(
            triv_tokenizer.tokenize(tok.strip('<>'))
        )
        embed = bart.encoder.embed_tokens.weight.data[char_idx[0]]
        for c_i in char_idx[1:]:
            embed += bart.encoder.embed_tokens.weight.data[c_i]
        embed /= len(char_idx)
        embed = embed.new_tensor(embed, requires_grad=True)
        bart.encoder.embed_tokens.weight.data[val] = embed

def init_cls_token_statistic(bart, dic_tok_id, triv_tokenizer, sentences):
    '''在bart模型中初始化特殊标记的嵌入向量'''
    num_tokens, _ = bart.encoder.embed_tokens.weight.shape
    bart.resize_token_embeddings(len(dic_tok_id)+num_tokens)
    last_w = {'word':'', 'tag':'o'}
    cls_margin_word = defaultdict(list)
    for sent in sentences:
        sent = sent + [last_w]
        w_ = {'word':'', 'tag':''}
        for w in sent:
            if w['tag'].startswith('b-'):
                '''实体开始的位置'''
                tag = "<<{}-s>>".format(w['tag'][2:])
                cls_margin_word[tag].append(w['word'])
                if w_['word']: cls_margin_word[tag].append(w_['word'])
                tag = "<<lab-s>>"
                cls_margin_word[tag].append(w['word'])
                if w_['word']: cls_margin_word[tag].append(w_['word'])
            if w['tag']=='o' and w_['tag'][:2] in ['i-','b-']: 
                '''实体结束的位置'''
                tag = "<<{}-e>>".format(w_['tag'][2:])
                cls_margin_word[tag].append(w_['word'])
                if w['word']: cls_margin_word[tag].append(w['word'])
                tag = '<<ent_end>>'
                cls_margin_word[tag].append(w_['word'])
                if w['word']: cls_margin_word[tag].append(w['word'])
            w_ = w
    cls_margin_word['<<lab-e>>'] = cls_margin_word['<<ent_end>>']
    for tok, val in dic_tok_id.items():
        char_idx = triv_tokenizer.convert_tokens_to_ids(
            triv_tokenizer.tokenize(''.join(cls_margin_word[tok]))
        )
        # print('train 131', tok, len(char_idx))
        embed = bart.encoder.embed_tokens.weight.data[char_idx].sum(dim=-2)
        # print(embed)
        # for c_i in char_idx[1:]:
        #     embed += bart.encoder.embed_tokens.weight.data[c_i]
        embed = embed/len(char_idx)
        embed = embed*torch.sqrt(2/torch.mul(embed,embed).sum())        
        # tmp = bart.encoder.embed_tokens.weight.data[char_idx[0]]
        # print('138', torch.mul(embed,embed).sum())
        # print(torch.mul(tmp,tmp).sum())
        # embed = embed.new_tensor(embed, requires_grad=True)
        embed = embed.clone().detach().requires_grad_(True)
        bart.encoder.embed_tokens.weight.data[val] = embed

def get_model_optim_sched(config, dic_tok_id):
    '''初始化模型、优化器、学习率函数'''
    device = config['device']
    model_path = config['model_path']
    if 'bart-base-chinese-cluecorpussmall' in model_path:
        from model.modeling_bart import BartModel
        # from transformers import BartModel
        bart = BartModel.from_pretrained(model_path).to(device)
    else:
        from model.modeling_bart import BartModel
        bart = BartModel.from_pretrained(model_path).to(device)

    triv_tokenizer = MyTokenizer.from_pretrained(model_path)
    if not config['margin_char_init']:
        init_cls_token_triv(bart, dic_tok_id, triv_tokenizer)
    else:
        file_path = os.path.join(config['dataset_dir'], 'train.train')
        sentences = parse_CoNLL_file(file_path)
        init_cls_token_statistic(bart, dic_tok_id, triv_tokenizer, sentences)
    # print('77', bart.decoder.embed_tokens.weight.data[21144])
    loss_fn = CrossEntropyLossWithMask()
    model = HiBart(bart, loss_fn, config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    total_step = config["total_steps"]
    batch_sched = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0.1*total_step,
        num_training_steps=total_step)
    epoch_sched = optim.lr_scheduler.MultiStepLR(
        optimizer, config["scheduler_step"], gamma=0.1)
    return model, optimizer, batch_sched, epoch_sched

def calib_pred(preds, end_pos, fold, targ=None):
    '''矫正解码的错误'''
    new_preds = []
    for ent in preds:
        for i in range(1,len(ent)):
            if ent[i] in end_pos and i<len(ent)-(fold-2):
                new_preds.append(ent[:i+fold-1])
                new_ent = ent[1:i]
                for j in range(1,len(new_ent)):
                    if new_ent[j]-new_ent[j-1]!=1:
                        new_preds = new_preds[:-1]
                        new_preds.append(ent[j+1:i+fold-1])
                        # '''只计算开始位置的正确率'''
                        # new_preds.append(ent[:2])
                        # new_preds.append(ent[j+1:j+3])
                        # print('train 190', ent)
                        # print('targ', targ)
                        break
                break
    return new_preds

def evaluate(config, model, loader):
    with torch.no_grad():
        model.eval()
        task_type = config['task_type']
        get_ents = get_semi_ents
        predicts, labels = [], []
        for multi_batch in loader:
            batch = multi_batch[task_type]
            # print('train 206', batch['prompt_pos_list'], task_type)
            # print(batch['targ_ents'][0])
            # print('dec_src_ids_bund', len(batch['dec_src_ids_bund']))
            # print(batch['dec_src_ids_bund'][0][0])
            # print(batch['dec_src_ids_bund'][1][0])
            # print('dec_targ_pos_bund', len(batch['dec_targ_pos_bund']))
            # print(batch['dec_targ_pos_bund'][0][0])
            # print(batch['dec_targ_pos_bund'][1][0])
            # print(batch['enc_mask'][0])
            dec_pred, flat_pred = model(
                batch['enc_src_ids'],
                batch['enc_src_len'],
                batch['enc_mask'],
                enc_attn_mask=batch['enc_attn_mask'],
                dec_src_ids_bund=batch['dec_src_ids_bund'],
                dec_src_pos_bund=batch['dec_src_pos_bund'],
                dec_mask_bund=batch['dec_mask_bund'],
                prompt_pos_list=batch['prompt_pos_list']
            )
            # pred = [get_targ_ents(p, rotate_pos_cls) for p in pred]
            # pred = [calib_pred(p, ent_end_pos) for p in pred]
            # print('train 219', pred[0])
            ent_pred = [get_ents(p, batch['prompt_pos_list'], task_type) for p in flat_pred]

            predicts += ent_pred
            labels += batch['targ_ents']
        model.train()
    return utils.micro_metrics(predicts, labels)

def use_model(model, batch):
    dec_pred, _ = model(
        batch['enc_src_ids'],
        batch['enc_src_len'],
        batch['enc_mask'],
        enc_attn_mask=batch['enc_attn_mask'],
        dec_src_ids_bund=batch['dec_src_ids_bund'],
        dec_src_pos_bund=batch['dec_src_pos_bund'],
        dec_mask_bund=batch['dec_mask_bund'],
        prompt_pos_list=batch['prompt_pos_list']
    )
    return dec_pred

def evaluate_2(config, model1, model2, loader):
    # config1 = config.copy()
    # config2 = config.copy()
    predicts = []
    labels = []
    with torch.no_grad():
        model1.eval()
        model2.eval()
        for multi_batch in loader:
            head_pred = use_model(model1, multi_batch['head'])
            tail_pred = use_model(model2, multi_batch['tail'])
            dec_src = multi_batch['head']['dec_src_pos_bund'][0]
            for i in range(len(head_pred)):
                ent_seq = get_ents_seq_from_semi_seq(
                    head_pred[i], tail_pred[i], dec_src[i],
                    config['rotate_pos_cls']
                )
                ents = get_targ_ents_2(
                    ent_seq, config['rotate_pos_cls']
                )
                predicts.append(ents)
            labels += multi_batch['cls']['targ_ents']
        model1.train()
        model2.train()
    return utils.micro_metrics(predicts, labels)

# def get_ents_seq_from_semi_seq(head_pred, tail_pred, multi_batch):
#     '''根据两个半边界序列得到完整的序列'''
#     head_pos = multi_batch['head']['prompt_pos_list']
#     tail_pos = multi_batch['tail']['prompt_pos_list']
#     dec_src = multi_batch['head']['dec_src_pos_bund'][0]
#     batch_ent_seq = []
#     for i in range(len(dec_src)):
#         ent_seq = []
#         for j in range(len(dec_src[i])):
#             if dec_src[i][j]==-1: continue
#             ent_seq.append(dec_src[i][j])
#             if tail_pred[i][j] in tail_pos:
#                 ent_seq.append(tail_pred[i][j])
#             if head_pred[i][j] in head_pos:
#                 ent_seq.append(head_pred[i][j])
#         batch_ent_seq.append(ent_seq)
#     return batch_ent_seq

def deal_pre_conf(model_path, tokenizer):
    pre_conf = PretrainedConfig.from_pretrained(model_path)
    pre_conf.tag_num = len(tokenizer.dic_tok_id)
    pre_conf.tag_num = 0
    pre_conf.save_pretrained(model_path)

def train(config):
    logger, OUTPUT_DIR = get_logger_dir(config)
    config['device'] = device
    # 初始化分词器、数据集和模型
    try:
        tokenizer = get_tokenizer(config)
        deal_pre_conf(config['model_path'], tokenizer)
        config['eos_id'] = tokenizer.eos_token_id
        config['pad_value'] = tokenizer.pad_token_id
        data_dealer = DataDealer(tokenizer, config)
        # rotate_pos_cls = data_dealer.rotate_pos_cls
        loaders = get_data_loader(config, data_dealer)
        train_loader, test_loader, valid_loader = loaders
        config["total_steps"] = config["epochs"] * len(train_loader)
        m_o_s = get_model_optim_sched(config, tokenizer.dic_tok_id)
        model, optimizer, batch_sched, epoch_sched = m_o_s
        logger("Init model.")
    except KeyboardInterrupt:
        logger("Interrupted.")
        logger.fp.close()
        os.system("rm -rf %s" % OUTPUT_DIR)
    except Exception as e:
        import traceback
        logger("Got exception.")
        logger.fp.close()
        os.system("rm -rf %s" % OUTPUT_DIR)
        print(traceback.format_exc())
    
    # 训练模型
    torch.set_printoptions(precision=6)
    try:
        logger("Begin training.")
        accum_loss = []
        best_f1 = 0.
        best_epoch = -1
        optimizer.zero_grad()
        step = 0
        denomin = 1
        if config['targ_self_sup']: denomin += 1
        
        task_list = ['head', 'tail']
        for epoch in range(config["epochs"]):
            model.train()
            stage = epoch % denomin

            for task_type in task_list:
                print(task_type)
                config['task_type'] = task_type
                if task_type=='head':
                    config['special_tok_pos'] = tokenizer.dic_hir_pos_cls[0]
                else:
                    config['special_tok_pos'] = tokenizer.dic_hir_pos_cls[1]
                for multi_batch in train_loader:
                    batch = multi_batch[task_type]
                    step += 1
                    batch = multi_batch[task_type]
                    enc_src_ids = batch['enc_src_ids']
                    enc_src_len = batch['enc_src_len']
                    enc_padding_mask = batch['enc_mask']
                    enc_attn_mask = batch['enc_attn_mask']
                    # print('train 308', batch['targ_ents'][0])
                    # print('train 309', enc_src_ids[0])
                    # print('train 310', batch['dec_src_ids_bund'][0][0])
                    # print('train 311', batch['dec_targ_pos_bund'][0][0])

                    loss, pred = model(
                        enc_src_ids,
                        enc_src_len,
                        enc_padding_mask,
                        enc_attn_mask=enc_attn_mask,
                        dec_src_ids_bund=batch['dec_src_ids_bund'],
                        dec_mask_bund=batch['dec_mask_bund'],
                        dec_targ_pos_bund=batch['dec_targ_pos_bund'],
                        prompt_pos_list=batch['prompt_pos_list'],
                        train_range=[stage]
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    if step % int(config["grad_accum_step"]) == 0:
                        optimizer.step()
                        batch_sched.step()
                        optimizer.zero_grad()
                    accum_loss.append(loss.item())
                    if step % int(config["show_loss_step"]) == 0:
                        # print('train 273', model.encoder.embed_tokens.weight.data[21144][5:10])
                        # print('train 273', model.encoder.embed_tokens.weight.data[21144].requires_grad)
                        mean_loss = sum(accum_loss) / len(accum_loss)
                        logger("Epoch %d, step %d / %d, loss = %.4f" % (
                            epoch+1, step, len(train_loader), mean_loss
                        ))
                        accum_loss = []
                epoch_sched.step()

                if epoch>=config['start_eval']:
                    valid_metrics = evaluate(config, model, valid_loader)
                    vep, ver, vef, vep1, ver1, vef1 = [m*100 for m in valid_metrics]
                    logger("Epoch %d, valid entity p=%.2f%%, r=%.2f%%, f=%.2f%%, p=%.2f%%, r=%.2f%%, f=%.2f%%" % (
                        epoch + 1, vep, ver, vef, vep1, ver1, vef1))
                    test_metrics = evaluate(config, model, test_loader)
                    tep, ter, tef, tep1, ter1, tef1 = [m*100 for m in test_metrics]
                    logger("Epoch %d, test entity p=%.2f%%, r=%.2f%%, f=%.2f%%, p=%.2f%%, r=%.2f%%, f=%.2f%%" % (
                        epoch + 1, tep, ter, tef, tep1, ter1, tef1))
                    if tef >= best_f1:
                        best_f1 = tef
                        best_epoch = epoch
                        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "snapshot.model"))
                        logger("Epoch %d, save model." % (epoch+1))
            
        logger("Best epoch %d, best entity f1: %.2f%%" % (best_epoch+1, best_f1))
        logger.fp.close()
    except KeyboardInterrupt:
        logger("Interrupted.")
        logger("Best epoch %d, best entity f1: %.2f%%" % (best_epoch+1, best_f1))
        logger.fp.close()
    except Exception as e:
        import traceback
        logger("Got exception.")
        logger.fp.close()
        print(traceback.format_exc())

def predict(config):
    config['device'] = device
    # 初始化分词器、数据集和模型
    tokenizer = get_tokenizer(config)
    config['eos_id'] = tokenizer.eos_token_id
    config['pad_value'] = tokenizer.pad_token_id
    # config['dic_hir_pos_cls'] = tokenizer.dic_hir_pos_cls
    data_dealer = DataDealer(tokenizer, config)
    # rotate_pos_cls = data_dealer.rotate_pos_cls
    loaders = get_data_loader(config, data_dealer)
    train_loader, test_loader, valid_loader = loaders
    config["total_steps"] = config["epochs"] * len(train_loader)
    model = get_model_optim_sched(config, tokenizer.dic_tok_id)[0]

    if config['task_type']=='head':
        config['special_tok_pos'] = tokenizer.dic_hir_pos_cls[0]
    else:
        config['special_tok_pos'] = tokenizer.dic_hir_pos_cls[1]
    state_dict_path = os.path.join(config["saved_path"], 'snapshot.model')
    model.load_state_dict(torch.load(state_dict_path))
    valid_metrics = evaluate(config, model, valid_loader)
    vep, ver, vef, vep1, ver1, vef1 = [m*100 for m in valid_metrics]
    print("valid entity p=%.2f%%, r=%.2f%%, f=%.2f%%, p=%.2f%%, r=%.2f%%, f=%.2f%%" % (
        vep, ver, vef, vep1, ver1, vef1))
    test_metrics = evaluate(config, model, test_loader)
    tep, ter, tef, tep1, ter1, tef1 = [m*100 for m in test_metrics]
    print("test entity p=%.2f%%, r=%.2f%%, f=%.2f%%, p=%.2f%%, r=%.2f%%, f=%.2f%%" % (
        tep, ter, tef, tep1, ter1, tef1))

def predict_2(config):
    config['device'] = device
    # 初始化分词器、数据集和模型
    tokenizer = get_tokenizer(config)
    config['eos_id'] = tokenizer.eos_token_id
    config['pad_value'] = tokenizer.pad_token_id
    config['rotate_pos_cls'] = tokenizer.rotate_pos_cls
    data_dealer = DataDealer(tokenizer, config)
    # rotate_pos_cls = data_dealer.rotate_pos_cls
    loaders = get_data_loader(config, data_dealer)
    train_loader, test_loader, valid_loader = loaders
    config["total_steps"] = config["epochs"] * len(train_loader)
    model1 = get_model_optim_sched(config, tokenizer.dic_tok_id)[0]
    model2 = get_model_optim_sched(config, tokenizer.dic_tok_id)[0]

    head_state_dict_path = os.path.join(config["head_saved_path"], 'snapshot.model')
    model1.load_state_dict(torch.load(head_state_dict_path))
    tail_state_dict_path = os.path.join(config["tail_saved_path"], 'snapshot.model')
    model1.load_state_dict(torch.load(tail_state_dict_path))

    valid_metrics = evaluate_2(config, model1, model2, valid_loader)
    vep, ver, vef, vep1, ver1, vef1 = [m*100 for m in valid_metrics]
    print("valid entity p=%.2f%%, r=%.2f%%, f=%.2f%%, p=%.2f%%, r=%.2f%%, f=%.2f%%" % (
        vep, ver, vef, vep1, ver1, vef1))
    test_metrics = evaluate_2(config, model1, model2, test_loader)
    tep, ter, tef, tep1, ter1, tef1 = [m*100 for m in test_metrics]
    print("test entity p=%.2f%%, r=%.2f%%, f=%.2f%%, p=%.2f%%, r=%.2f%%, f=%.2f%%" % (
        tep, ter, tef, tep1, ter1, tef1))

    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test', default=0, type=int, choices=[0,1])
par_args = parser.parse_args()

if __name__=='__main__':
    if not par_args.test:
        config_path = 'config.json'
        with open(config_path, encoding="utf-8") as fp: config = json.load(fp)
        config['config_path'] = config_path
        config['dataset_dir'] = os.path.join(config['data_dir'], config['dataset'])
        torch.autograd.set_detect_anomaly(True)
        with torch.autograd.detect_anomaly():
            train(config)
    else:
        saved_path = "/data1/nzw/model_saved/HiBart1/weibo_single tail"
        config_path = os.path.join(saved_path, 'config.json')
        with open(config_path, encoding="utf-8") as fp: config = json.load(fp)
        config['saved_path'] = saved_path
        config['config_path'] = config_path
        config['dataset_dir'] = os.path.join(config['data_dir'], config['dataset'])
        config['task_type'] = 'tail'
        predict(config)

        config["head_saved_path"] = "/data1/nzw/model_saved/HiBart1/weibo_single head"
        config["tail_saved_path"] = "/data1/nzw/model_saved/HiBart1/weibo_single tail"
        predict_2(config)
