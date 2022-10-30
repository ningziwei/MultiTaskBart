import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset import flat_sequence

class HiDecoder(nn.Module):
    def __init__(self, decoder, args):
        super(HiDecoder, self).__init__()
        self.decoder = decoder
        self.args = args
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_mask', causal_mask.float())
        hidden_size = decoder.embed_tokens.weight.size(1)
        # self.decoder_mlp = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.Dropout(0.3)
        # )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(0.3)
        self.lstm = nn.LSTM(768, 384, 1, bidirectional=True)
    
    def forward(
        self, enc_output, src_embed,
        enc_src_len, enc_padding_mask, 
        dec_src_ids, dec_padding_mask, dec_src_len=None,
        prompt_pos_list=None
    ):
        '''
        预测一次输出，得到token是每个候选token的概率
        enc_output: bsz*max_len*emb_dim
        enc_src_len: bsz*1
        '''
        # 过解码器，得到隐藏状态
        causal_mask = self.causal_mask[
            :dec_src_ids.size(1), :dec_src_ids.size(1)]
        dec_state_dic = self.decoder(
            input_ids=dec_src_ids,
            encoder_hidden_states=enc_output,
            encoder_padding_mask=enc_padding_mask,
            decoder_padding_mask=dec_padding_mask,
            decoder_causal_mask=causal_mask,
            return_dict=True
        )
        dec_output = dec_state_dic.last_hidden_state
        # dec_output = self.dropout_layer(dec_output)
        dec_output = self.self_attn_layer_norm(dec_output)
        # dec_output = self.decoder_mlp(dec_output)
        
        # 用双向lstm处理decoder的输出向量
        if self.args['use_lstm']:
            packed_embs = pack_padded_sequence(
                dec_output, dec_src_len.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, (_, _) = self.lstm(packed_embs)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            dec_output = self.dropout_layer(lstm_out)

        # 初始化预测结果，bsz*dec_out_max_len*enc_out_max_len
        '''可能要补+1'''
        logits = dec_output.new_full(
            list(dec_output.size()[:2])+[enc_output.size(1)],
            fill_value=-1e32)
        
        # enc_src_embed = self.decoder.embed_tokens(enc_src_ids)
        # enc_src_embed = self.dropout_layer(enc_src_embed)
        # enc_output = self.encoder_mlp(enc_output)
        # enc_output = self.dropout_layer(enc_output)
        # src_embed = (enc_src_embed + enc_output)/2
        word_scores = torch.einsum('blh,bnh->bln', dec_output, src_embed)
        
        if self.args['static_eos']:
            # 静态结束符，用词表中的结束符嵌入向量作为目标
            eos_id = self.args['eos_id']
            eos_emb = self.decoder.embed_tokens.weight[eos_id:eos_id+1]
            eos_scores = F.linear(dec_output, self.dropout_layer(eos_emb))
        else:
            # 动态结束符，用结束符的编码向量作为目标
            eos_scores = word_scores[range(len(enc_output)),:,enc_src_len-1]
            eos_scores = eos_scores.unsqueeze(-1)
        
        '''
        word_scores: bsz*max_dec_len*max_enc_len
        enc_padding_mask: bsz*1*max_enc_len
        dec_padding_mask: bsz*max_dec_len*1
        '''
        # 结束标记的得分不算在word_score中
        enc_padding_mask[range(len(enc_padding_mask)), enc_src_len-1] = 0
        enc_padding_mask = enc_padding_mask.eq(0)
        enc_padding_mask = enc_padding_mask.unsqueeze(1)
        word_scores = word_scores.masked_fill(enc_padding_mask, -1e32)
        # dec_padding_mask = dec_padding_mask.eq(0).unsqueeze(-1)
        # word_scores = word_scores.masked_fill(dec_padding_mask, -1e32)

        # print('hi_bart 78', logits.shape, eos_scores.shape, word_scores.shape)
        # print('hi_bart 78', logits, eos_scores, word_scores)
        logits[:,:,1:2] = eos_scores
        logits[:,:,:1] = word_scores[:,:,0:1]
        # print('hi_bart 94', prompt_pos_list)
        if prompt_pos_list is not None:
            # print('hi_bart 94', prompt_pos_list)
            # print(logits.shape, word_scores.shape)
            min_p = min(prompt_pos_list)
            max_p = max(prompt_pos_list)
            p_num = len(prompt_pos_list)
            logits[:,:,min_p:max_p+1] = word_scores[:,:,min_p-1:max_p]
            # print('99', logits[:,:,min_p:max_p+1])
            # logits[:,:,max_p+1+1:] = word_scores[:,:,1+p_num:-1]
            logits[:,:,2+p_num*2:] = word_scores[:,:,1+p_num*2:-1]
        else:
            logits[:,:,2:] = word_scores[:,:,1:-1]
        # print('logits', logits)
        return logits

class HiBart(nn.Module):
    def __init__(self, bart, loss_fn, args):
        super(HiBart, self).__init__()
        self.encoder = bart.encoder
        self.loss_fn = loss_fn
        self.args = args

        self.dropout_layer = nn.Dropout(0.5)
        hidden_size = self.encoder.embed_tokens.weight.size(1)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3)
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.hi_decoder = HiDecoder(bart.decoder, args)
        

    def forward(
        self, enc_src_ids, enc_src_len, 
        enc_padding_mask, enc_attn_mask=None,
        dec_src_ids_bund=None, dec_src_pos_bund=None,
        dec_mask_bund=None, dec_targ_pos_bund=None,
        prompt_pos_list = None, train_range=range(3),
    ):
        '''
        enc_src_ids: batch_size*enc_max_len
        enc_src_len: batch_size*1
        enc_padding_mask: batch_size*enc_max_len
        enc_attn_mask: batch_size*enc_max_len*enc_max_len
        dec_src_ids_bund: 3*batch_size*dec_max_len
        dec_mask_bund: 3*batch_size*dec_max_len
        dec_targ_pos_bund: 3*batch_size*dec_max_len
        分阶段解码经过三次解码器，最后的loss是三次loss的和
        '''
        enc_state_dic = self.encoder(
            input_ids=enc_src_ids, attention_mask=enc_padding_mask, 
            return_dict=True, output_hidden_states=True,
            enc_attn=enc_attn_mask
        )
        enc_output = enc_state_dic.last_hidden_state
        '''src_embed 用于在decoder输出部分计算相似度'''
        enc_src_embed = self.encoder.embed_tokens(enc_src_ids)
        enc_src_embed = self.dropout_layer(enc_src_embed)
        enc_output_mlp = self.encoder_mlp(enc_output)
        src_embed = (enc_src_embed + enc_output_mlp)/2
        src_embed = self.self_attn_layer_norm(src_embed)
        # enc_src_embed = self.encoder.embed_tokens(enc_src_ids)
        # enc_src_embed = self.self_attn_layer_norm(enc_src_embed)
        # enc_output_mlp = self.encoder_mlp(enc_output)
        # src_embed = (enc_src_embed + enc_output_mlp)/2
        # src_embed = self.self_attn_layer_norm(src_embed)
        '''
        训练过程是batch_size*3*dec_max_len，三条都直接用到
        预测过程是batch_size*1*dec_max_len，只用第一条，后面的边生成边解码
        '''
        if dec_targ_pos_bund is not None:
            '''训练过程，三段训练依次进行，各loss求和后一起更新'''
            batch_loss = 0
            for i in train_range:
                dec_src_ids = dec_src_ids_bund[i]
                dec_padding_mask = dec_mask_bund[i]
                # print('163', dec_padding_mask.shape)
                # print(dec_padding_mask[:,-5:])
                dec_src_len = dec_padding_mask.sum(dim=-1)
                # print(dec_src_len)
                dec_targ_pos = dec_targ_pos_bund[i]
                logits = self.hi_decoder(
                    enc_output, src_embed,
                    enc_src_len, enc_padding_mask,
                    dec_src_ids, dec_padding_mask, dec_src_len,
                    prompt_pos_list
                )
                batch_loss += self.loss_fn(
                    logits, dec_targ_pos, dec_padding_mask)
            batch_pred = torch.argmax(logits, dim=-1)
            return batch_loss, batch_pred
        else:
            '''
            预测过程，执行后解码，解码结果再给到decoder
            return:
              batch_pred: decoder直接的预测结果，pad部分是-1
              flat_pred: 将dec_src跟batch_pred拉平后的结果，没有pad
            '''
            dec_src_ids = dec_src_ids_bund[0]
            dec_src_pos = dec_src_pos_bund[0]
            dec_padding_mask = dec_mask_bund[0]
            # print('174', dec_src_ids[0])
            # print('175', dec_src_pos[0])
            eval_num = len(dec_src_ids_bund)

            if self.args['targ_self_sup']:
                eval_num -= 1
            if self.args['src_self_sup']:
                eval_num -= 1
                dec_src_ids = dec_src_ids_bund[1]
                dec_src_pos = dec_src_pos_bund[1]
                dec_padding_mask = dec_mask_bund[1]
            eval_range = range(eval_num)
            for i in eval_range:
                dec_src_len = dec_padding_mask.sum(dim=-1)
                logits = self.hi_decoder(
                    enc_output, src_embed,
                    enc_src_len, enc_padding_mask,
                    dec_src_ids, dec_padding_mask, dec_src_len,
                    prompt_pos_list=prompt_pos_list
                )
                batch_pred = torch.argmax(logits, dim=-1)
                dec_src_ids = dec_src_ids.masked_fill(dec_padding_mask.eq(0), -1)
                batch_pred = batch_pred.masked_fill(dec_padding_mask.eq(0), -1)
                # print('hi_bart 183', batch_pred[0])
                # print('184', enc_src_ids[0])
                # print('185', dec_src_ids[0])
                # print('186', dec_src_pos[0])
                # print('187', dic_hir_pos_cls[i])
                dec_src_ids, dec_src_pos, dec_padding_mask, flat_pred = flat_sequence(
                    batch_pred.cpu().numpy(),
                    batch_enc_src_ids=enc_src_ids.cpu().numpy(), 
                    batch_dec_src_ids=dec_src_ids.cpu().numpy(),
                    batch_dec_src_pos=dec_src_pos.cpu().numpy(),
                    dic_pos_cls=prompt_pos_list,
                    pad_value=self.args['pad_value'],
                    device=self.args['device']
                )
                dec_src_len = dec_padding_mask.sum(dim=-1)
            return batch_pred.cpu().numpy(), flat_pred

