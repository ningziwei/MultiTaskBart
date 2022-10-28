import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossWithMask(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWithMask, self).__init__()
    
    def test_func(self):
        minmum = -1e32
        bsz = 2
        dec_num = 6
        size_ = [bsz,dec_num]
        enc_num = 4

        logits = torch.torch.randn(size_ + [enc_num])

        enc_mask = torch.randint(0,2,[bsz,enc_num])
        enc_mask = enc_mask.unsqueeze(1)
        logits = logits.masked_fill(enc_mask, minmum)

        dec_mask = torch.randint(0,2,size_)
        dec_mask = dec_mask.eq(0)
        logits = logits.masked_fill(dec_mask.unsqueeze(-1), minmum)

        dec_targ_pos = torch.randint(0,enc_num,size_)
        dec_targ_pos = dec_targ_pos.masked_fill(dec_mask, -100)
        print('losses 28', logits.shape, dec_targ_pos.shape)
        print(logits)
        print(dec_targ_pos)
        loss_tgt = F.cross_entropy(
            input=logits.transpose(1, 2), 
            target=dec_targ_pos)
        print(loss_tgt)

    def test_nan(self, pred, tgt_tokens, mask):
        mask = mask.eq(0)
        one_hot_tgt = F.one_hot(tgt_tokens, pred.shape[-1]).float()
        one_hot_tgt = one_hot_tgt.masked_fill(mask.unsqueeze(-1), 0)
        pred_util = pred - torch.max(pred, 2).values.unsqueeze(-1)
        logsoftmax = pred_util-torch.log(torch.sum(torch.exp(pred_util), dim = -1).reshape(pred.shape[0], -1, 1))
        nllloss_tgt = -torch.sum(one_hot_tgt*logsoftmax)/torch.sum(1-mask.float())
        # print('nllloss_tgt', nllloss_tgt)
        # print('pred', pred.shape, pred[0])
        # print('tgt_tokens', tgt_tokens.shape, tgt_tokens[0])
        # print('mask', mask.shape, mask[0])
        # print('one_hot_tgt', one_hot_tgt.shape, one_hot_tgt[0])
        # print('pred_util', pred_util.shape, pred_util[0])
        # print('logsoftmax', logsoftmax.shape, logsoftmax[0])
        # print('50', torch.sum(1-mask.float()))
        tmp = pred
        for i in range(tmp.size(0)):
            # print('52', i)
            for j in range(tmp.size(1)):
                for k in range(tmp.size(2)):
                    if tmp[i][j][k]>100 or tmp[i][j][k]<-100 or torch.isnan(tmp[i][j][k]):
                        print(i,j,k)
                        print(tmp[i][j])
                        print(pred_util[i][j])
                        return

    def forward(self, logits, dec_targ_pos, dec_mask):
        '''
        logits: bsz*max_dec_len*max_enc_len
        dec_targ_pos: bsz*max_dec_len
        dec_mask: bsz*max_dec_len
        cross_entropy的ignore_index是-100，即目标值为-100时
        忽略其对应的loss
        '''
        # self.test_nan(logits, dec_targ_pos, dec_mask)
        dec_mask = dec_mask.eq(0)
        dec_targ_pos = dec_targ_pos.masked_fill(dec_mask, -100)
        loss_tgt = F.cross_entropy(
            input=logits.transpose(1, 2), 
            target=dec_targ_pos)
        # print('losses 55', loss_tgt)
        return loss_tgt



