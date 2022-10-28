import time

class Logger(object):
    def __init__(self, fp=None):
        self.fp = fp

    def __call__(self, string, end='\n'):
        curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        new_string = '[%s] ' % curr_time + string
        print(new_string, end=end)
        if self.fp is not None:
            self.fp.write('%s%s' % (new_string, end))

def other_metrics(predicts, labels):
    true_count, predict_count, gold_count = 0, 0, 0
    for pred_entity, gold_entity in zip(predicts, labels):
        '''头边界的正确率'''
        pred_entity = [p[:2] for p in pred_entity]
        gold_entity = [g[:2] for g in gold_entity]
        for e in pred_entity:
            if e in gold_entity:
                true_count += 1
        predict_count += len(pred_entity)
        gold_count += len(gold_entity)
    ep = true_count / max(predict_count, 1)
    er = true_count / gold_count
    ef = 2 * ep * er / max((ep + er), 0.0001)
    print("头边界 p=%.2f%%, r=%.2f%%, f=%.2f%%" % (ep, er, ef))
    
    # true_count, predict_count, gold_count = 0, 0, 0
    # for pred_entity, gold_entity in zip(predicts, labels):
    #     '''尾边界正确率'''
    #     pred_entity = [p[-3:-1] for p in pred_entity]
    #     gold_entity = [g[-3:-1] for g in gold_entity]
    #     for e in pred_entity:
    #         if e in gold_entity:
    #             true_count += 1
    #     predict_count += len(pred_entity)
    #     gold_count += len(gold_entity)
    # ep = true_count / max(predict_count, 1)
    # er = true_count / gold_count
    # ef = 2 * ep * er / max((ep + er), 0.0001)
    # print("尾边界 p=%.2f%%, r=%.2f%%, f=%.2f%%" % (ep, er, ef))

def micro_metrics(predicts, labels):
    '''计算预测指标'''
    # other_metrics(predicts, labels)
    true_count, predict_count, gold_count = 0, 0, 0
    for pred_entity, gold_entity in zip(predicts, labels):
        '''带分类的正确率'''
        # print('utils 55', pred_entity)
        # print('utils 56', gold_entity)
        for e in pred_entity:
            if e in gold_entity:
                true_count += 1
        predict_count += len(pred_entity)
        gold_count += len(gold_entity)
    wep = true_count / max(predict_count, 1)
    wer = true_count / gold_count
    wef = 2 * wep * wer / max((wep + wer), 0.0001)

    true_count, predict_count, gold_count = 0, 0, 0
    for pred_entity, gold_entity in zip(predicts, labels):
        '''边界的正确率'''
        for e in pred_entity:
            if e in gold_entity:
                true_count += 1
        predict_count += len(pred_entity)
        gold_count += len(gold_entity)
    ep = true_count / max(predict_count, 1)
    er = true_count / gold_count
    ef = 2 * ep * er / max((ep + er), 0.0001)
    return wep, wer, wef, ep, er, ef