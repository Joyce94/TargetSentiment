import argparse
import data.config as config
from data.vocab import Data

import model_att.vanilla_att as vanilla_att
import model_att.context_att_b as context_att_b
import model_att.context_att_gate as context_att_gate
import model_att.context_att_gate_b as context_att_gate_b
import train_att
import torch
import random
import numpy as np
import time

if __name__ == '__main__':
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(666)
    torch.backends.cudnn.enabled = False

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default='/home/song/TargetSentiment/vanilla/examples/config.cfg')
    argparser.add_argument('--use-cuda', default=True)
    argparser.add_argument('--static', default=False, help='fix the embedding')
    argparser.add_argument('--add-char', default=False, help='add char feature')
    argparser.add_argument('--metric', default='exact', help='choose from [exact, binary, proportional]')
    argparser.add_argument('--model', default='vanilla', help='choose from [vanilla, context_att, context_att_gate]')

    # args = argparser.parse_known_args()
    args = argparser.parse_args()
    config = config.Configurable(args.config_file)

    data = Data()
    data.number_normalized = False
    data.static = args.static
    data.add_char = args.add_char
    data.use_cuda = args.use_cuda
    data.metric = args.metric

    test_time = time.time()
    train_insts, train_insts_index = data.get_instance(config.train_file, config.run_insts, config.shrink_feature_thresholds, char=args.add_char)
    print('test getting train_insts time: ', time.time()-test_time)
    if not args.static:
        data.fix_alphabet()
    dev_insts, dev_insts_index = data.get_instance(config.dev_file, config.run_insts, config.shrink_feature_thresholds, char=args.add_char)
    print('test getting dev_insts time: ', time.time() - test_time)

    data.fix_alphabet()
    test_insts, test_insts_index = data.get_instance(config.test_file, config.run_insts, config.shrink_feature_thresholds, char=args.add_char)
    print('test getting test_insts time: ', time.time() - test_time)

    if config.pretrained_wordEmb_file != '':
        data.norm_word_emb = False
        data.build_word_pretrain_emb(config.pretrained_wordEmb_file, config.word_dims)
    if config.pretrained_charEmb_file != '':
        data.norm_char_emb = False
        data.build_char_pretrain_emb(config.pretrained_charEmb_file, config.char_dims)

    if args.model == 'context_att':
        # model_att = context_att.Context_att(config, data)
        model_att = context_att_b.Context_att(config, data)
    elif args.model == 'vanilla':
        model_att = vanilla_att.Vanilla_att(config, data)
    elif args.model == 'context_att_gate':
        model_att = context_att_gate_b.Context_att_gate(config, data)

    print('test building model time: ', time.time() - test_time)

    if data.use_cuda: model_att = model_att.cuda()


    train_att.train_att(train_insts, train_insts_index, dev_insts, dev_insts_index, test_insts, test_insts_index, model_att, config, data)
