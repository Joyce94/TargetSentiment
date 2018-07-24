import torch
import torch.nn as nn
import time
import random
import numpy as np
import data.utils as utils
import data.vocab as vocab
import torch.nn.functional as F


def to_scalar(vec):
    return vec.view(-1).data.tolist()[0]

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_att(train_insts, train_insts_index, dev_insts, dev_insts_index, test_insts, test_insts_index, model_att, config, params):
    print('training...')
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters_att = filter(lambda p: p.requires_grad, model_att.parameters())
    # optimizer = torch.optim.Adam(params=parameters, lr=config.learning_rate, weight_decay=config.decay)
    # optimizer_att = torch.optim.Adam(params=parameters_att, lr=config.learning_rate, weight_decay=config.decay)
    # optimizer_att = torch.optim.SGD(params=parameters_att, lr=config.learning_rate, momentum=0.9)
    optimizer_att = torch.optim.SGD(params=parameters_att, lr=config.learning_rate, momentum=0.9, weight_decay=config.decay)
    #optimizer_att = torch.optim.Adagrad(params=parameters_att, lr=config.learning_rate, weight_decay=config.decay)

    best_micro_f1 = float('-inf')
    best_macro_f1 = float('-inf')

    for epoch in range(config.maxIters):
        start_time = time.time()
        model_att.train()
        train_insts, train_insts_index = utils.random_data(train_insts, train_insts_index)
        epoch_loss_e = 0
        train_buckets, train_labels_raw, train_category_raw, train_target_start, train_target_end = params.generate_batch_buckets(config.train_batch_size, train_insts_index, char=params.add_char)

        for index in range(len(train_buckets)):
            batch_length = np.array([np.sum(mask) for mask in train_buckets[index][-1]])
            fea_v, label_v, mask_v, length_v, target_v, start_v, end_v = utils.patch_var(train_buckets[index], batch_length.tolist(), train_category_raw[index], train_target_start[index], train_target_end[index], params)
            model_att.zero_grad()

            if mask_v.size(0) != config.train_batch_size:
                model_att.hidden = model_att.init_hidden(mask_v.size(0), config.lstm_layers)
            else:
                model_att.hidden = model_att.init_hidden(config.train_batch_size, config.lstm_layers)

            logit = model_att.forward(fea_v, batch_length.tolist(), start_v, end_v, label_v)
            # print(target_v)
            loss_e = F.cross_entropy(logit, target_v)
            loss_e.backward()
            optimizer_att.step()
            epoch_loss_e += to_scalar(loss_e)

            # # nn.utils.clip_grad_norm(model.parameters(), config.clip_grad)
        # print('\nepoch is {}, average loss_c is {} '.format(epoch, (epoch_loss_c / config.train_batch_size)))
        print('\nepoch is {}, average loss_e is {} '.format(epoch, (epoch_loss_e / (config.train_batch_size * len(train_buckets)))))
        # update lr
        adjust_learning_rate(optimizer_att, config.learning_rate / (1 + (epoch + 1) * config.decay))
        print('Dev...')
        # dev_acc, dev_f1 = eval_att_single(dev_insts, dev_insts_index, model_att, config, params)
        # if dev_f1 > best_f1 or dev_acc > best_acc:
        #     if dev_f1 > best_f1: best_f1 = dev_f1
        #     if dev_acc > best_acc: best_acc = dev_acc
        #     print('\nTest...')
        #     test_acc, test_f1 = eval_att_single(test_insts, test_insts_index, model_att, config, params)
        # print('now, best fscore is {}, best accuracy is {}'.format(best_f1, best_acc))
        dev_micro_fscore, dev_macro_fscore = eval_att(dev_insts, dev_insts_index, model_att, config, params)
        if dev_micro_fscore > best_micro_f1:
            best_micro_f1 = dev_micro_fscore
            # print('\nTest...')
            # test_acc = eval_att(test_insts, test_insts_index, model_att, config, params)
        if dev_macro_fscore > best_macro_f1:
            best_macro_f1 = dev_macro_fscore
        print('now, best micro fscore is {}%, best macro fscore is {}%'.format(best_micro_f1, best_macro_f1))


def eval_att(insts, insts_index, model, config, params):
    model.eval()
    # model_e.eval()
    insts, insts_index = utils.random_data(insts, insts_index)
    buckets, labels_raw, categorys_raw, target_start, target_end = params.generate_batch_buckets(len(insts), insts_index, char=params.add_char)

    size = len(insts)
    batch_length = np.array([np.sum(mask) for mask in buckets[0][-1]])
    fea_v, label_v, mask_v, length_v, target_v, start_v, end_v = utils.patch_var(buckets[0], batch_length.tolist(), categorys_raw, target_start, target_end, params)
    target_v = target_v.squeeze(0)
    start_v = start_v.squeeze(0)
    end_v = end_v.squeeze(0)

    if mask_v.size(0) != config.test_batch_size:
        model.hidden = model.init_hidden(mask_v.size(0), config.lstm_layers)
    else:
        model.hidden = model.init_hidden(config.test_batch_size, config.lstm_layers)

    logit = model.forward(fea_v, batch_length.tolist(), start_v, end_v, label_v)
    micro_fscore, macro_fscore = calc_fscore(logit, target_v, size, params)

    return micro_fscore, macro_fscore


def calc_fscore(logit, target_v, size, params):
    max_index = torch.max(logit, dim=1)[1].view(target_v.size())
    rel_list = [[], [], []]
    pre_list = [[], [], []]
    corrects_list = [[], [], []]

    # print(params.category_alphabet.id2word)     # ['neutral', 'negative', 'positive']
    # print(params.category_alphabet.word2id)     # OrderedDict([('neutral', 0), ('negative', 1), ('positive', 2)])
    corrects = 0
    for x in range(max_index.size(0)):
        target_id = int(to_scalar(target_v[x]))
        rel_list[target_id].append(1)
        predict_id = int(to_scalar(max_index[x]))

        if predict_id == target_id:
            corrects += 1
            corrects_list[target_id].append(1)
        pre_list[predict_id].append(1)
    c_list = [len(ele) for ele in corrects_list]
    r_list = [len(ele) for ele in rel_list]
    p_list = [len(ele) for ele in pre_list]
    # assert (torch.max(logit, 1)[1].view(target_v.size()).data == target_v.data).sum() == corrects

    # recall = [float(x) / r_list[id] * 100.0 for id, x in enumerate(c_list)]
    recall = []
    for id, x in enumerate(c_list):
        if r_list[id] != 0:
            temp = float(x)/r_list[id]
        else:
            temp = 0
        recall.append(temp)
    # precision = [float(x) / p_list[id] * 100.0 for id, x in enumerate(c_list)]
    precision = []
    for id, x in enumerate(c_list):
        if p_list[id] != 0:
            temp = float(x)/p_list[id]
        else:
            temp = 0
        precision.append(temp)
    f_score = []
    for idx, p in enumerate(precision):
        if p + recall[idx] == 0:
            f_score.append(0.0)
        else:
            f_score.append(2 * p * recall[idx] / (p + recall[idx]))
    for i in range(len(c_list)):
        category_cur = params.category_alphabet.id2word[i]
        # print(category_cur)
        print('{}: precision: {:.4f}%, recall: {}%, fscore: {}% ({}/{}/{})'.format(category_cur, precision[i]*100.0, recall[i]*100.0, f_score[i]*100.0, c_list[i], p_list[i], r_list[i]))

    micro_fscore = float(corrects) / size * 100.0
    macro_fscore = (f_score[0] + f_score[1] + f_score[2]) / 3 * 100.0
    print('\nEvaluation - micro fscore: {:.4f}%({}/{}), macro fscore: {}% \n'.format(micro_fscore, corrects, size, macro_fscore))

    return micro_fscore, macro_fscore







