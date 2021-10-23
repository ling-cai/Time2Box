#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import Query2box
from dataloader import *
from tensorboardX import SummaryWriter
import time
import pickle
import collections
import math
import torch.optim.lr_scheduler as lr_scheduler

from utils import * 

import ray
from ray import tune

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=100000, type=int)
    
    # parser.add_argument('--save_checkpoint_steps', default=5000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--model_save_step', default=100, type=int, help='save model in every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--ntimestamp', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('--geo', default='vec', type=str, help='vec or box')
    parser.add_argument('--print_on_screen', action='store_true')
    
    parser.add_argument('--task', default='1c.2c.3c.2i.3i', type=str)
    parser.add_argument('--stepsforpath', type=int, default=0)

    parser.add_argument('--offset_deepsets', default='vanilla', type=str, help='inductive or vanilla or min')
    parser.add_argument('--offset_use_center', action='store_true')
    parser.add_argument('--center_deepsets', default='vanilla', type=str, help='vanilla or attention or mean')
    parser.add_argument('--center_use_offset', action='store_true')
    parser.add_argument('--entity_use_offset', action='store_true')
    parser.add_argument('--att_reg', default=0.0, type=float)
    parser.add_argument('--off_reg', default=0.0, type=float)
    parser.add_argument('--att_tem', default=1.0, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gamma2', default=0, type=float)
    parser.add_argument('--train_onehop_only', action='store_true')
    parser.add_argument('--center_reg', default=0.0, type=float, help='alpha in the paper')
    parser.add_argument('--time_smooth_weight', default=0.0, type=float, help='weight for time smoother in the paper')
    parser.add_argument('--time_smoother', type=str, default='L2', help='regularizater used in the paper')
    parser.add_argument('--use_fixed_time_fun', action='store_true', help='pure learning or given the func form')
    parser.add_argument('--use_separate_relation_embedding', action='store_true', help='whether to use a separate relation embedding to consider time')
    parser.add_argument('--bn', default='no', type=str, help='no or before or after')
    parser.add_argument('--n_att', type=int, default=1)
    parser.add_argument('--activation', default='relu', type=str, help='relu or none or softplus')
    parser.add_argument('--act_time', default='none', type=str, help='periodical or non-periodical activation function')
    parser.add_argument('--label', default='test', type=str, help='checkpoint label-- label whether this is the best one')

    # whether to use samples for statements with full intervals
    parser.add_argument('--use_relation_time', action='store_true', help='use another relation when creating temporal statements')
    
    parser.add_argument('--use_one_sample', action='store_true', help='convert full interval to point-in-time, only sample one from the interval')
    parser.add_argument('--use_two_sample', action='store_true', help='still as full interval but the start and end is sampled from the true interval')
    parser.add_argument('--add_inverse', action='store_true', help='determine whether to add tuples with inverse relation')
    parser.add_argument('--add_hard_neg', action='store_true', help='determine whether to add hard negative samples')
    parser.add_argument('--double_point_in_time', action='store_true', help='determine whether to repeat time in point-in-time')
    parser.add_argument('--enumerate_time', action='store_true', help='determine whether to repeat time in point-in-time')
    parser.add_argument('--negative_sample_types', default='tail-batch', type=str)

    parser.add_argument('--time_score_weight', default=0.1, type=float)
    parser.add_argument('--num_time_negatives', default=0, type=int)

    parser.add_argument('--flag_use_weighted_partial_interval', action='store_true')

    parser.add_argument('--predict_o', action='store_true')
    parser.add_argument('--predict_t', action='store_true')
    parser.add_argument('--predict_r', action='store_true')

    return parser.parse_args(args)

def override_config(args): #! may update here
    '''s
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, args.data_path.split('/')[-2], args.model, args.label, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    # args.valid_steps = 20000
    # args.time_smooth_weight = 0.00001
    
def save_model(model, optimizer, save_variable_list, args, before_finetune=False, best_model=False):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    if best_model:
        if not os.path.exists(os.path.join(args.save_path, 'best_model')):
            os.makedirs(os.path.join(args.save_path, 'best_model'))
        save_path = os.path.join(args.save_path, 'best_model')
    else:
        save_path = args.save_path

    argparse_dict = vars(args)
    with open(os.path.join(save_path, 'config.json' if not before_finetune else 'config_before.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_path, 'checkpoint' if not before_finetune else 'checkpoint_before')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(save_path, 'entity_embedding' if not before_finetune else 'entity_embedding_before'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(save_path, 'relation_embedding' if not before_finetune else 'relation_embedding_before'), 
        relation_embedding
    )
    if model.use_fixed_time_fun:
        time_frequency = model.time_frequency.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'time_frequency' if not before_finetune else 'time_frequency_before'), 
            time_frequency
        )

        time_shift = model.time_shift.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'time_shift' if not before_finetune else 'time_shift_before'), 
            time_shift
        )
    else:
        time_embedding = model.time_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'time_embedding' if not before_finetune else 'time_embedding_before'), 
            time_embedding
        )


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        tag = 'predict_o'
        if args.predict_t:
            tag = 'predict_t'
        elif args.predict_r:
            tag = 'predict_r'

        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test%s.log' % (tag))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

from argparse import Namespace
def parameter_tune_main(config):
    ## convert to namespace
    # print('get here')
    while True:
        main(args, mode='TUNE')

        
def main(args, mode=''):
    if isinstance(args, dict):
        args = Namespace(**args)

    set_global_seed(args.seed)
    # args.test_batch_size = 4
    assert args.bn in ['no', 'before', 'after']
    assert args.n_att >= 1 and args.n_att <= 3

    if args.geo == 'box':
        assert 'Box' in args.model
    elif args.geo == 'vec':
        assert 'Box' not in args.model
        
    if args.train_onehop_only:
        assert '1c' in args.task
        args.center_deepsets = 'mean'
        if args.geo == 'box':
            args.offset_deepsets = 'min'

    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    # if args.do_train and args.save_path is None:
    #     raise ValueError('Where do you want to save your trained model?')

    cur_time = parse_time()
    print ("overide save string.")
    if args.task == '1c':
        args.stepsforpath = 0
    else:
        assert args.stepsforpath <= args.max_steps
    
    args.save_path = 'logs/%s%s/%s/%s/%s'%(args.data_path.split('/')[-1], args.geo, args.data_path.split('/')[-2], args.model, args.label)
    writer = SummaryWriter(args.save_path)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    set_logger(args)

    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

        if 'wikidata' in args.data_path.lower() or 'yago' in args.data_path.lower():
            ntimestamp = int(entrel[2].split(' ')[-1])
            args.ntimestamp = ntimestamp
    
    args.nentity = nentity
    if args.add_inverse:
        logging.info('add inverse: True')
        args.nrelation = nrelation*2
        args.ntimestamp = args.ntimestamp*2
    else:
        args.nrelation = nrelation
        logging.info('add inverse: False')

    # print('double_point_in_time')
    
    logging.info('Geo: %s' % args.geo)
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % args.nentity)
    logging.info('#relation: %d' % args.nrelation)
    logging.info('#timestamps: %d' % args.ntimestamp)
    logging.info('#max steps: %d' % args.max_steps)
    

    tasks = args.task.split('.')
    # args.negative_sample_types = args.negative_sample_types.split('.')

    # args.model_save_step = args.model_save_step if args.model_save_step >= args.valid_steps else args.valid_steps
    
    train_ans = dict()
    valid_ans = dict()
    valid_ans_hard = dict()
    test_ans = dict()
    test_ans_hard = dict()

    # args.data_path = args.data_path.replace('wikidata_toy_expanded', 'wikidata_toy_new')
    ## num_triples in an epoch
    num_triples_per_task = {}
    ## in some datasets, they do not have 1c
    ## in that case, we use 1c-expanded but do not include the result in the final result.
    if '1c' in tasks:
        ## always use 1c-expanded when training
        with open('%s/train_triples_1c.pkl'%args.data_path, 'rb') as handle:
            train_triples = pickle.load(handle)

        with open('%s/valid_triples_1c.pkl'%args.data_path, 'rb') as handle:
            valid_triples = pickle.load(handle)
        # if len(valid_triples) == 0: ## no atemporal statements existing in the KB; used for monitoring the loss
        #     with open('%s/valid_triples_1c_expanded.pkl'%args.data_path, 'rb') as handle:
        #         valid_triples = pickle.load(handle)
       
        with open('%s/test_triples_1c.pkl'%args.data_path, 'rb') as handle:
            test_triples = pickle.load(handle)

        with open('%s/train_ans_1c.pkl'%args.data_path, 'rb') as handle:
            train_ans_1 = pickle.load(handle)
        with open('%s/valid_ans_1c.pkl'%args.data_path, 'rb') as handle:
            valid_ans_1 = pickle.load(handle)
        with open('%s/test_ans_1c.pkl'%args.data_path, 'rb') as handle:
            test_ans_1 = pickle.load(handle)

        with open('%s/valid_ans_1c_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_1_hard = pickle.load(handle)
        with open('%s/test_ans_1c_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_1_hard = pickle.load(handle)

        valid_ans_hard.update(valid_ans_1_hard)
        test_ans_hard.update(test_ans_1_hard)
        train_ans.update(train_ans_1)
        valid_ans.update(valid_ans_1)
        test_ans.update(test_ans_1)

        num_triples_per_task['1c'] = len(train_triples)

    if '2i' in tasks:
        with open('%s/train_triples_2i.pkl'%args.data_path, 'rb') as handle:
            train_triples_2i = pickle.load(handle)
        with open('%s/train_ans_2i.pkl'%args.data_path, 'rb') as handle:
            train_ans_2i = pickle.load(handle)

        with open('%s/valid_triples_2i_begin.pkl'%args.data_path, 'rb') as handle:
            valid_triples_2i_begin = pickle.load(handle)
        with open('%s/valid_triples_2i_end.pkl'%args.data_path, 'rb') as handle:
            valid_triples_2i_end = pickle.load(handle)
        with open('%s/valid_ans_2i.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2i = pickle.load(handle)
        with open('%s/valid_ans_2i_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2i_hard = pickle.load(handle)

        with open('%s/test_triples_2i_begin.pkl'%args.data_path, 'rb') as handle:
            test_triples_2i_begin = pickle.load(handle)
        with open('%s/test_triples_2i_end.pkl'%args.data_path, 'rb') as handle:
            test_triples_2i_end = pickle.load(handle)
        with open('%s/test_ans_2i.pkl'%args.data_path, 'rb') as handle:
            test_ans_2i = pickle.load(handle)       
        with open('%s/test_ans_2i_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_2i_hard = pickle.load(handle)

        num_triples_per_task['2i'] = len(train_triples_2i)

    with open('%s/train_ans_2i.pkl'%args.data_path, 'rb') as handle:
        train_ans_2i = pickle.load(handle)
    with open('%s/valid_ans_2i.pkl'%args.data_path, 'rb') as handle:
        valid_ans_2i = pickle.load(handle)
    with open('%s/valid_ans_2i_hard.pkl'%args.data_path, 'rb') as handle:
        valid_ans_2i_hard = pickle.load(handle)
    with open('%s/test_ans_2i.pkl'%args.data_path, 'rb') as handle:
        test_ans_2i = pickle.load(handle)       
    with open('%s/test_ans_2i_hard.pkl'%args.data_path, 'rb') as handle:
        test_ans_2i_hard = pickle.load(handle)


    valid_ans_hard.update(valid_ans_2i_hard)
    test_ans_hard.update(test_ans_2i_hard)
    train_ans.update(train_ans_2i)
    valid_ans.update(valid_ans_2i)
    test_ans.update(test_ans_2i)

        

    if '3i-2i' in tasks: ## the answer set is in '2i'; begin_only interval (3)end_only_interval
        with open('%s/train_triples_3i_2i.pkl'%args.data_path, 'rb') as handle:
            train_triples_3i_2i = pickle.load(handle)
            num_triples_per_task['3i-2i'] = len(train_triples_3i_2i)

        with open('%s/test_triples_3i_2i.pkl'%args.data_path, 'rb') as handle:
            test_triples_3i_2i = pickle.load(handle)

        with open('%s/valid_triples_3i_2i.pkl'%args.data_path, 'rb') as handle:
            valid_triples_3i_2i = pickle.load(handle)

        num_triples_per_task['3i-2i'] = len(train_triples_3i_2i)

    if '3i' in tasks:
        '''
        one case: full intervals with accurate begin date and end date 
        '''
        with open('%s/train_triples_3i.pkl'%args.data_path, 'rb') as handle:
            train_triples_3i = pickle.load(handle)
            num_triples_per_task['3i'] = len(train_triples_3i)

        if args.use_two_sample:
            with open('%s/train_ans_3i.pkl'%args.data_path, 'rb') as handle:
                train_ans_3i = pickle.load(handle)
                train_ans.update(train_ans_3i)

        with open('%s/valid_triples_3i.pkl'%args.data_path, 'rb') as handle:
            valid_triples_3i = pickle.load(handle)
        with open('%s/valid_ans_3i.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3i = pickle.load(handle)
        with open('%s/valid_ans_3i_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3i_hard = pickle.load(handle)

        with open('%s/test_triples_3i.pkl'%args.data_path, 'rb') as handle:
            test_triples_3i = pickle.load(handle)
        with open('%s/test_ans_3i.pkl'%args.data_path, 'rb') as handle:
            test_ans_3i = pickle.load(handle)
        with open('%s/test_ans_3i_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_3i_hard = pickle.load(handle)

        valid_ans_hard.update(valid_ans_3i_hard)
        test_ans_hard.update(test_ans_3i_hard)
        
        valid_ans.update(valid_ans_3i)
        test_ans.update(test_ans_3i)

    if 'time-batch' in args.negative_sample_types:
        with open('%s/ans_t.pkl'%args.data_path, 'rb') as handle:
            ans_t = pickle.load(handle)
    else:
        ans_t = None

    if args.predict_t:
        with open('%s/test_ans_t.pkl'%args.data_path, 'rb') as handle:
            test_ans_t = pickle.load(handle) 
        with open('%s/valid_ans_t.pkl'%args.data_path, 'rb') as handle:
            valid_ans_t = pickle.load(handle)
    else:
        test_ans_t = None
        valid_ans_t = None
        
    ## get the share of each task used in train
    num_triples_per_epoch = sum([num for num in num_triples_per_task.values()])

    if '1c' in tasks:
        logging.info('#train: %d' % len(train_triples))
        logging.info('#valid: %d' % len(valid_triples))
        logging.info('#test: %d' % len(test_triples))

    # if '1c-t' in tasks:
    #     logging.info('#train-t: %d' % len(train_triples_t))
    #     logging.info('#valid-t: %d' % len(valid_triples_t))
    #     # logging.info('#test-t: %d' % len(test_triples_t))
    
    if '2i' in tasks:
        logging.info('#train_2i: %d' % len(train_triples_2i))
        logging.info('#valid_2i_begin: %d' % len(valid_triples_2i_begin))
        logging.info('#valid_2i_end: %d' % len(valid_triples_2i_end))
        logging.info('#test_2i_begin: %d' % len(test_triples_2i_begin))
        logging.info('#test_2i_end: %d' % len(test_triples_2i_end))

    if '3i-2i' in tasks:
        logging.info('#train_3i_2i: %d' % len(train_triples_3i_2i))
        logging.info('#valid_3i_2i: %d' % len(valid_triples_3i_2i))
        logging.info('#test_3i_2i: %d' % len(test_triples_3i_2i))
    
    if '3i' in tasks:
        logging.info('#train_3i: %d' % len(train_triples_3i))
        logging.info('#valid_3i: %d' % len(valid_triples_3i))
        logging.info('#test_3i: %d' % len(test_triples_3i))

    ## in order to make balance between different sub datasets; merge 2i to 3i or 3i-2i
    if args.double_point_in_time and args.use_two_sample:
        raise NotImplementedError
    elif args.double_point_in_time:
        logging.info('merge 2i to 3i')
        train_triples_3i.extend(train_triples_2i)
    else:
        logging.info('merge 2i and 3i-2i')
        train_triples_3i_2i.extend(train_triples_2i)

    if args.do_train:
        # Set training dataloader iterator
        if '1c' in tasks: ## here we do not have time information 
            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, train_ans_1, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainDataset.collate_fn
            )
            train_iterator = SingledirectionalOneShotIterator(train_dataloader_tail, train_triples[0][-1])

        # if '2i' in tasks:
        #     train_dataloader_2i_tail = DataLoader(
        #         TrainInterDataset(train_triples_2i, nentity, nrelation, args.negative_sample_size, train_ans_2i, 'tail-batch', add_hard_neg=args.add_hard_neg), 
        #         batch_size=args.batch_size,
        #         shuffle=True, 
        #         num_workers=max(1, args.cpu_num),
        #         collate_fn=TrainInterDataset.collate_fn
        #         )
            
        #     train_iterator_2i = SingledirectionalOneShotIterator(train_dataloader_2i_tail, train_triples_2i[0][-1])

        if '3i-2i' in tasks:
            # if 'tail-batch' in args.negative_sample_types:
            train_dataloader_3i_2i_tail = DataLoader(
                TrainInterDataset(train_triples_3i_2i, nentity, nrelation, ntimestamp, args.negative_sample_size, train_ans_2i, mode=args.negative_sample_types,  ans_t=ans_t, use_one_sample=args.use_one_sample, use_two_sample=args.use_two_sample, add_hard_neg=args.add_hard_neg, double_point_in_time=args.double_point_in_time, num_time_negatives=args.num_time_negatives), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn
                )
            if args.double_point_in_time:
                qtype = '2-3-inter'
            else:
                qtype = train_triples_2i[0][-1]
            train_iterator_3i_2i = SingledirectionalOneShotIterator(train_dataloader_3i_2i_tail, qtype)

            # if 'time-batch' in args.negative_sample_types:
            #     train_dataloader_3i_2i_time = DataLoader(
            #         TrainInterDataset(train_triples_3i_2i, nentity, nrelation, ntimestamp, args.negative_sample_size//8, ans_t, 'time-batch', add_hard_neg=args.add_hard_neg, double_point_in_time=args.double_point_in_time), 
            #         batch_size=args.batch_size,
            #         shuffle=True, 
            #         num_workers=max(1, args.cpu_num),
            #         collate_fn=TrainInterDataset.collate_fn
            #         )
            #     if args.double_point_in_time:
            #         qtype = '2-3-inter'
            #     else:
            #         qtype = train_triples_2i[0][-1]
            #     train_iterator_3i_2i_time = SingledirectionalOneShotIterator(train_dataloader_3i_2i_time, qtype)


        if '3i' in tasks: # there are three datasets that should be considered
            if args.use_one_sample:
                qtype = '2-inter'
                # train_ans_here = train_ans_2i
            else:
                # train_ans_here = train_ans_3i
                qtype = train_triples_3i[0][-1]

            # if 'tail-batch' in args.negative_sample_types:
            train_dataloader_3i_tail = DataLoader(
                TrainInterDataset(train_triples_3i, nentity, nrelation, ntimestamp, args.negative_sample_size, train_ans_2i if args.use_one_sample else train_ans, mode = args.negative_sample_types,  ans_t=ans_t, use_one_sample=args.use_one_sample, use_two_sample=args.use_two_sample, add_hard_neg=args.add_hard_neg, double_point_in_time=args.double_point_in_time, num_time_negatives=args.num_time_negatives), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn
            )
            train_iterator_3i = SingledirectionalOneShotIterator(train_dataloader_3i_tail, qtype)

            # if 'time-batch' in args.negative_sample_types:
            #     train_dataloader_3i_time = DataLoader(
            #     TrainInterDataset(train_triples_3i, nentity, nrelation, ntimestamp, args.negative_sample_size//8, ans_t, 'time-batch', args.use_one_sample, args.use_two_sample, args.add_hard_neg), 
            #     batch_size=args.batch_size,
            #     shuffle=True, 
            #     num_workers=max(1, args.cpu_num),
            #     collate_fn=TrainInterDataset.collate_fn
            #     )
            #     train_iterator_3i_time = SingledirectionalOneShotIterator(train_dataloader_3i_time, qtype)


    if args.time_smooth_weight != 0.0:
        if args.time_smoother == 'Lambda3':
            time_reg = Lambda3(args.time_smooth_weight)
        elif args.time_smoother == 'L2':
            time_reg =  L2(args.time_smooth_weight)
    else:
        time_reg = None

    query2box = Query2box(
        model_name=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        ntimestamp=args.ntimestamp,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        writer=writer,
        geo=args.geo,
        cen=args.center_reg,
        offset_deepsets = args.offset_deepsets, ## method used for aggregating off_sets of boxes
        center_deepsets = args.center_deepsets, ## method used for aggregating centers of boxes
        offset_use_center = args.offset_use_center, ## whether to use center information when aggregating the offsets
        center_use_offset = args.center_use_offset, ## whether to use offset information when aggregating the centers
        att_reg = args.att_reg,
        off_reg = args.off_reg,
        att_tem = args.att_tem,
        euo = args.entity_use_offset, ## whether to treat entities as boxes as well
        gamma2 = args.gamma2,
        bn = args.bn, # when to use batch normalization; no, before, after
        nat = args.n_att, ## num of layers in an attention module
        activation = args.activation,
        act_time=args.act_time,
        use_fixed_time_fun = args.use_fixed_time_fun,
        time_reg = time_reg,
        use_separate_relation_embedding = args.use_separate_relation_embedding,
        use_relation_time=args.use_relation_time
    )
    
    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in query2box.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        query2box = query2box.cuda()
            
    # Set training configuration
    current_learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, query2box.parameters()), 
        lr=current_learning_rate
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2, verbose=True)  # mrr tracking
        # if args.warm_up_steps:
        #     warm_up_steps = args.warm_up_steps
        # else:
        #     warm_up_steps = args.max_steps // 2
    # args.data_path = args.data_path.replace('wikidata_toy_new', 'wikidata_toy_expanded')
    if args.init_checkpoint:
        # Restore model from checkpoint directory
        
        try:
            checkpoint = torch.load(os.path.join(args.init_checkpoint, args.data_path.split('/')[-2], args.model, args.label, 'best_model', 'checkpoint'))
            logging.info('Loading checkpoint %s...' % os.path.join(args.init_checkpoint, args.data_path.split('/')[-2], args.model, args.label, 'best_model', 'checkpoint'))
        except:
            checkpoint = torch.load(os.path.join(args.init_checkpoint, args.data_path.split('/')[-2], args.model, args.label, 'checkpoint'))
            logging.info('Loading checkpoint %s...' % os.path.join(args.init_checkpoint, args.data_path.split('/')[-2], args.model, args.label, 'checkpoint'))

        init_step = checkpoint['step']
        if 'best_valid_mrrm' in checkpoint:
            best_valid_mrrm = checkpoint['best_valid_mrrm']
            last_valid_mrrm = checkpoint['curr_valid_mrrm']
        else:
            best_valid_mrrm = 0.0
            last_valid_mrrm = 0.0
        query2box.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            # current_learning_rate = 0.0001
            # warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ## change learning_rate
            for g in optimizer.param_groups:
                g['lr'] = 0.001
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2, verbose=True)
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
        best_valid_mrrm = 0.0
        last_valid_mrrm = 0.0

    step = init_step 

    logging.info('all param setting used in the model\n %s' % args)
    logging.info('init_step = %d' % init_step)
    logging.info('best valid mrr = %f' % best_valid_mrrm)
    logging.info('last_valid_mrrm = %f' % last_valid_mrrm)
    # logging.info('number of triples in training = %d' % num_triples_per_epoch)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %f' % current_learning_rate)
    # logging.info('batch_size = %d' % args.batch_size)
    # logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    # logging.info('hidden_dim = %d' % args.hidden_dim)
    # logging.info('gamma = %f' % args.gamma)
    # logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    # if args.negative_adversarial_sampling:
    #     logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    def evaluate_test():
        average_metrics = collections.defaultdict(list)
        average_c_metrics = collections.defaultdict(list)
        average_c2_metrics = collections.defaultdict(list)
        average_i_metrics = collections.defaultdict(list)
        average_2i_metrics = collections.defaultdict(list)
        average_ex_metrics = collections.defaultdict(list)
        average_u_metrics = collections.defaultdict(list)
        total_number_triples = 0
        total_number_triples_i = 0
        total_number_triples_c = 0

        ## save rank results
        if not os.path.exists(args.save_path+'/rank_result/'):
            os.makedirs(args.save_path+'/rank_result/')

        checkpoint = torch.load(os.path.join(args.save_path, 'best_model', 'checkpoint'))

        query2box.load_state_dict(checkpoint['model_state_dict'])

        if '2i' in tasks:
            if len(test_triples_2i_begin)>0:
                metrics = query2box.test_step(query2box, test_triples_2i_begin, test_ans, test_ans_hard, args, '2i-begin', test_ans_t)
                # if args.predict_o:
                log_metrics('Test only-begin', step, metrics)
                num_triples = len(test_triples_2i_begin)
                total_number_triples += num_triples
                total_number_triples_i += num_triples
                for metric in metrics:
                    writer.add_scalar('Test/Test_2i_begin_'+metric, metrics[metric], step)
                    average_metrics[metric].append(metrics[metric]*num_triples)
                    average_i_metrics[metric].append(metrics[metric]*num_triples)
                    average_2i_metrics[metric].append(metrics[metric]*num_triples)
            if len(test_triples_2i_end)>0:
                metrics = query2box.test_step(query2box, test_triples_2i_end, test_ans, test_ans_hard, args, '2i-end', test_ans_t)
                # if args.predict_o:
                log_metrics('Test only-end', step, metrics)
                num_triples = len(test_triples_2i_end)
                total_number_triples += num_triples
                total_number_triples_i += num_triples
                for metric in metrics:
                    writer.add_scalar('Test/Test_2i_end_'+metric, metrics[metric], step)
                    average_metrics[metric].append(metrics[metric]*num_triples)
                    average_i_metrics[metric].append(metrics[metric]*num_triples)
                    average_2i_metrics[metric].append(metrics[metric]*num_triples)

        if '3i' in tasks:
            metrics = query2box.test_step(query2box, test_triples_3i, test_ans, test_ans_hard, args, '3i', test_ans_t)
            if args.predict_o:
                log_metrics('Test full time', step, metrics)
                num_triples = len(test_triples_3i)
                total_number_triples += num_triples
                total_number_triples_i += num_triples
                for metric in metrics:
                    writer.add_scalar('Test/Test_3i_'+metric, metrics[metric], step)
                    average_metrics[metric].append(metrics[metric]*num_triples)
                    average_i_metrics[metric].append(metrics[metric]*num_triples)

        if '3i-2i' in tasks:
            metrics = query2box.test_step(query2box, test_triples_3i_2i, test_ans, test_ans_hard, args, '3i-2i', test_ans_t)
            # if args.predict_o:
            log_metrics('Test point time', step, metrics)
            num_triples = len(test_triples_3i_2i)
            total_number_triples += num_triples
            total_number_triples_i += num_triples
            for metric in metrics:
                writer.add_scalar('Test/Test_3i_2i'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric]*num_triples)
                average_i_metrics[metric].append(metrics[metric]*num_triples)

        # if '2c' in tasks:
        #     metrics = query2box.test_step(query2box, test_triples_2, test_ans, test_ans_hard, args)
        #     log_metrics('Test 2c', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Test_2c_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_c_metrics[metric].append(metrics[metric])
        #         average_c2_metrics[metric].append(metrics[metric])
        # if '3c' in tasks:
        #     metrics = query2box.test_step(query2box, test_triples_3, test_ans, test_ans_hard, args)
        #     log_metrics('Test 3c', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Test_3c_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_c_metrics[metric].append(metrics[metric])
        #         average_c2_metrics[metric].append(metrics[metric])
        
        if '1c' in tasks and args.predict_o:
            if len(test_triples) != 0:
                metrics = query2box.test_step(query2box, test_triples, test_ans, test_ans_hard, args, '1c')
                log_metrics('Test no time', step, metrics)
                num_triples = len(test_triples)
                total_number_triples += num_triples
                total_number_triples_c += num_triples
                for metric in metrics:
                    writer.add_scalar('Test/Test_1c_'+metric, metrics[metric], step)
                    average_metrics[metric].append(metrics[metric]*num_triples)
                    average_c_metrics[metric].append(metrics[metric]*num_triples)

        # if 'ci' in tasks:
        #     metrics = query2box.test_step(query2box, test_triples_ci, test_ans, test_ans_hard, args)
        #     log_metrics('Test ci', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Test_ci_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_ex_metrics[metric].append(metrics[metric])
        # if 'ic' in tasks:
        #     metrics = query2box.test_step(query2box, test_triples_ic, test_ans, test_ans_hard, args)
        #     log_metrics('Test ic', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Test_ic_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_ex_metrics[metric].append(metrics[metric])
        # if '2u' in tasks:
        #     metrics = query2box.test_step(query2box, test_triples_2u, test_ans, test_ans_hard, args)
        #     log_metrics('Test 2u', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Test_2u_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_u_metrics[metric].append(metrics[metric])
        # if 'uc' in tasks:
        #     metrics = query2box.test_step(query2box, test_triples_uc, test_ans, test_ans_hard, args)
        #     log_metrics('Test uc', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Test_uc_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_u_metrics[metric].append(metrics[metric])
        for metric in average_metrics:
            writer.add_scalar('Test/Test_average_'+metric, np.sum(average_metrics[metric])/total_number_triples, step)
            log_metrics('Test average_metrics_', step, {metric: np.sum(average_metrics[metric])/total_number_triples})
        for metric in average_c_metrics:
            writer.add_scalar('Test/Test_average_no_time_'+metric, np.sum(average_c_metrics[metric])/total_number_triples_c, step)
            log_metrics('Test average_no_time_metrics_', step, {metric: np.sum(average_c_metrics[metric])/total_number_triples_c})
        # for metric in average_c2_metrics:
        #     writer.add_scalar('Test/Test_average_c2_'+metric, np.sum(average_c2_metrics[metric]/tot), step)
        for metric in average_i_metrics:
            writer.add_scalar('Test/Test_average_i_'+metric, np.sum(average_i_metrics[metric])/total_number_triples_i, step)
            log_metrics('Test average_with_time_metrics_', step, {metric: np.sum(average_i_metrics[metric])/total_number_triples_i})

        for metric in average_2i_metrics:
            writer.add_scalar('Test/Test_average_partial_'+metric, np.sum(average_2i_metrics[metric])/(len(test_triples_2i_begin)+len(test_triples_2i_end)), step)
            log_metrics('Test average_partial_time_metrics_', step, {metric: np.sum(average_2i_metrics[metric])/(len(test_triples_2i_begin)+len(test_triples_2i_end))})
        # for metric in average_u_metrics:
        #     writer.add_scalar('Test/Test_average_u_'+metric, np.mean(average_u_metrics[metric]), step)
        # for metric in average_ex_metrics:
        #     writer.add_scalar('Test/Test_average_ex_'+metric, np.mean(average_ex_metrics[metric]), step)

    def print_named_parameters():
        for k, v in query2box.named_parameters():
            print(k, v[:10])

    def evaluate_val(tasks):
        average_metrics = collections.defaultdict(list)
        average_c_metrics = collections.defaultdict(list)
        average_c2_metrics = collections.defaultdict(list)
        average_i_metrics = collections.defaultdict(list)
        average_2i_metrics = collections.defaultdict(list)
        average_ex_metrics = collections.defaultdict(list)
        average_u_metrics = collections.defaultdict(list)

        total_number_triples = 0
        total_number_triples_i = 0
        total_number_triples_c = 0

        # ## save rank results
        if not os.path.exists(args.save_path+'/rank_result/'):
            os.makedirs(args.save_path+'/rank_result/')

        # checkpoint = torch.load(os.path.join(args.save_path, 'best_model', 'checkpoint'))

        # query2box.load_state_dict(checkpoint['model_state_dict'])

        # if '2i-1c' in tasks:
        #     metrics = query2box.test_step(query2box, valid_triples_2i_1c, valid_ans, valid_ans_hard, args)
        #     log_metrics('Valid 2i 1c', step, metrics)
        #     num_triples = len(valid_triples_2i_1c)
        #     total_number_triples += num_triples
        #     total_number_triples_c += num_triples
        #     for metric in metrics:
        #         writer.add_scalar('Valid_2i_1c_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric]*num_triples)
        #         average_i_metrics[metric].append(metrics[metric]*num_triples)
        if '2i' in tasks:   
            if len(valid_triples_2i_begin) > 0: 
                metrics = query2box.test_step(query2box, valid_triples_2i_begin, valid_ans, valid_ans_hard, args, '2i-begin-valid' if args.predict_t else '', valid_ans_t)
                log_metrics('Valid only-begin time', step, metrics)
                num_triples = len(valid_triples_2i_begin)
                total_number_triples += num_triples
                total_number_triples_i += num_triples
                for metric in metrics:
                    writer.add_scalar('valid/Valid_2i_begin_'+metric, metrics[metric], step) 
                    average_metrics[metric].append(metrics[metric]*num_triples)
                    average_i_metrics[metric].append(metrics[metric]*num_triples)
                    average_2i_metrics[metric].append(metrics[metric]*num_triples)
            if len(valid_triples_2i_end) > 0:
                metrics = query2box.test_step(query2box, valid_triples_2i_end, valid_ans, valid_ans_hard, args, '2i-end-valid' if args.predict_t else '', valid_ans_t)
                log_metrics('Valid only-end time', step, metrics)
                num_triples = len(valid_triples_2i_end)
                total_number_triples += num_triples
                total_number_triples_i += num_triples
                for metric in metrics:
                    writer.add_scalar('valid/Valid_2i_end_'+metric, metrics[metric], step)
                    average_metrics[metric].append(metrics[metric]*num_triples)
                    average_i_metrics[metric].append(metrics[metric]*num_triples)
                    average_2i_metrics[metric].append(metrics[metric]*num_triples)

        if '3i-2i' in tasks:    
            metrics = query2box.test_step(query2box, valid_triples_3i_2i, valid_ans, valid_ans_hard, args, '3i-2i-valid' if args.predict_t else '', valid_ans_t)
            log_metrics('Valid point time', step, metrics)
            num_triples = len(valid_triples_3i_2i)
            total_number_triples += num_triples
            total_number_triples_i += num_triples
            for metric in metrics:
                writer.add_scalar('valid/Valid_3i_2i_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric]*num_triples)
                average_i_metrics[metric].append(metrics[metric]*num_triples)

        # if '3i-1c' in tasks:
        #     metrics = query2box.test_step(query2box, valid_triples_3i_1c, valid_ans, valid_ans_hard, args)
        #     log_metrics('Valid 3i 1c', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('valid/Valid_3i_1c_'+metric, metrics[metric]*num_triples, step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_i_metrics[metric].append(metrics[metric])

        if '3i' in tasks:
            metrics = query2box.test_step(query2box, valid_triples_3i, valid_ans, valid_ans_hard, args, '3i-valid' if args.predict_t else '', valid_ans_t)
            log_metrics('Valid full time', step, metrics)
            num_triples = len(valid_triples_3i)
            total_number_triples += num_triples
            total_number_triples_i += num_triples
            for metric in metrics:
                writer.add_scalar('valid/Valid_3i'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric]*num_triples)
                average_i_metrics[metric].append(metrics[metric]*num_triples)

        # if '2c' in tasks:
        #     metrics = query2box.test_step(query2box, valid_triples_2, valid_ans, valid_ans_hard, args)
        #     log_metrics('Valid 2c', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Valid_2c_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_c_metrics[metric].append(metrics[metric])
        #         average_c2_metrics[metric].append(metrics[metric])

        # if '3c' in tasks:
        #     metrics = query2box.test_step(query2box, valid_triples_3, valid_ans, valid_ans_hard, args)
        #     log_metrics('Valid 3c', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Valid_3c_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_c_metrics[metric].append(metrics[metric])
        #         average_c2_metrics[metric].append(metrics[metric])
        
        if '1c' in tasks:
            if len(valid_triples) != 0 and args.predict_o:
                metrics = query2box.test_step(query2box, valid_triples, valid_ans, valid_ans_hard, args)
                # assert len(valid_triples) != 0 ## always
                # tag = 'expanded' if len(test_triples) == 0 else ''
                log_metrics('Valid no time', step, metrics)
                num_triples = len(valid_triples)
                total_number_triples += num_triples
                total_number_triples_c += num_triples

                for metric in metrics:
                    writer.add_scalar('valid/Valid_1c_' + metric, metrics[metric], step)
                    average_metrics[metric].append(metrics[metric]*num_triples)
                    average_c_metrics[metric].append(metrics[metric]*num_triples)

        # if '1c-t' in tasks:
        #     metrics = query2box.test_step(query2box, valid_triples_t, valid_ans, valid_ans_hard, args)
        #     log_metrics('Valid 1c t', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Valid_1c_t_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_c_metrics[metric].append(metrics[metric])

        # if 'ci' in tasks:
        #     metrics = query2box.test_step(query2box, valid_triples_ci, valid_ans, valid_ans_hard, args)
        #     log_metrics('Valid ci', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Valid_ci_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_ex_metrics[metric].append(metrics[metric])
        # if 'ic' in tasks:
        #     metrics = query2box.test_step(query2box, valid_triples_ic, valid_ans, valid_ans_hard, args)
        #     log_metrics('Valid ic', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Valid_ic_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_ex_metrics[metric].append(metrics[metric])
        # if '2u' in tasks:
        #     metrics = query2box.test_step(query2box, valid_triples_2u, valid_ans, valid_ans_hard, args)
        #     log_metrics('Valid 2u', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Valid_2u_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_u_metrics[metric].append(metrics[metric])
        # if 'uc' in tasks:
        #     metrics = query2box.test_step(query2box, valid_triples_uc, valid_ans, valid_ans_hard, args)
        #     log_metrics('Valid uc', step, metrics)
        #     for metric in metrics:
        #         writer.add_scalar('Valid_uc_'+metric, metrics[metric], step)
        #         average_metrics[metric].append(metrics[metric])
        #         average_u_metrics[metric].append(metrics[metric])
        for metric in average_metrics:
            writer.add_scalar('valid/Valid_average_'+metric, np.sum(average_metrics[metric])/total_number_triples, step)
        for metric in average_c_metrics:
            writer.add_scalar('valid/Valid_average_c_'+metric, np.sum(average_c_metrics[metric])/total_number_triples_c, step)
        # for metric in average_c2_metrics:
        #     writer.add_scalar('Valid_average_c2_'+metric, np.mean(average_c2_metrics[metric]), step)
        for metric in average_i_metrics:
            writer.add_scalar('valid/Valid_average_i_'+metric, np.sum(average_i_metrics[metric])/total_number_triples_i, step)
        # for metric in average_u_metrics:
        #     writer.add_scalar('Valid_average_u_'+metric, np.mean(average_u_metrics[metric]), step)
        # for metric in average_ex_metrics:
        #     writer.add_scalar('Valid_average_ex_'+metric, np.mean(average_ex_metrics[metric]), step)
        for metric in average_2i_metrics:
            writer.add_scalar('valid/Valid_average_partial_'+metric, np.sum(average_2i_metrics[metric])/(len(valid_triples_2i_begin)+len(valid_triples_2i_end)), step)

        return np.sum(average_metrics['MRRm_new'])/total_number_triples
    
    def evaluate_train():
        average_metrics = collections.defaultdict(list)
        average_c_metrics = collections.defaultdict(list)
        average_c2_metrics = collections.defaultdict(list)
        average_i_metrics = collections.defaultdict(list)
        if '2i' in tasks:
            metrics = query2box.test_step(query2box, train_triples_2i, train_ans, train_ans, args)
            log_metrics('train 2i', step, metrics)
            for metric in metrics:
                writer.add_scalar('train/train_2i_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_i_metrics[metric].append(metrics[metric])

        if '3i' in tasks:
            # metrics = query2box.test_step(query2box, train_triples_3i, train_ans, train_ans, args)
            # log_metrics('train 3i', step, metrics)
            # for metric in metrics:
            #     writer.add_scalar('train_3i_'+metric, metrics[metric], step)
            #     average_metrics[metric].append(metrics[metric])
            #     average_i_metrics[metric].append(metrics[metric])

            metrics = query2box.test_step(query2box, train_triples_3i_1c, train_ans, train_ans, args)
            log_metrics('train 3i 1c', step, metrics)
            for metric in metrics:
                writer.add_scalar('train_3i_1c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_i_metrics[metric].append(metrics[metric])

            # metrics = query2box.test_step(query2box, train_triples_3i_begin, train_ans, train_ans, args, interval_type='begin-only')
            # log_metrics('train 3i begin', step, metrics)
            # for metric in metrics:
            #     writer.add_scalar('train_3i_begin_'+metric, metrics[metric], step)
            #     average_metrics[metric].append(metrics[metric])
            #     average_i_metrics[metric].append(metrics[metric])

            # metrics = query2box.test_step(query2box, train_triples_3i_end, train_ans, train_ans, args, interval_type='end-only')
            # log_metrics('train 3i end', step, metrics)
            # for metric in metrics:
            #     writer.add_scalar('train_3i_end_'+metric, metrics[metric], step)
            #     average_metrics[metric].append(metrics[metric])
            #     average_i_metrics[metric].append(metrics[metric])

        if '2c' in tasks:
            metrics = query2box.test_step(query2box, train_triples_2, train_ans, train_ans, args)
            log_metrics('train 2c', step, metrics)
            for metric in metrics:
                writer.add_scalar('train_2c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
                average_c2_metrics[metric].append(metrics[metric])
        if '3c' in tasks:
            metrics = query2box.test_step(query2box, train_triples_3, train_ans, train_ans, args)
            log_metrics('train 3c', step, metrics)
            for metric in metrics:
                writer.add_scalar('train_3c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
                average_c2_metrics[metric].append(metrics[metric])
        if '1c' in tasks:
            metrics = query2box.test_step(query2box, train_triples, train_ans, train_ans, args)
            log_metrics('train 1c', step, metrics)
            for metric in metrics:
                writer.add_scalar('train_1c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])

        if '1c-t' in tasks:
            metrics = query2box.test_step(query2box, train_triples_t, train_ans, train_ans, args)
            log_metrics('train 1c-t', step, metrics)
            for metric in metrics:
                writer.add_scalar('train_1c_t_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])

        for metric in average_metrics:
            writer.add_scalar('train_average_'+metric, np.mean(average_metrics[metric]), step)
        for metric in average_c_metrics:
            writer.add_scalar('train_average_c_'+metric, np.mean(average_c_metrics[metric]), step)
        for metric in average_c2_metrics:
            writer.add_scalar('train_average_c2_'+metric, np.mean(average_c2_metrics[metric]), step)
        for metric in average_i_metrics:
            writer.add_scalar('train_average_i_'+metric, np.mean(average_i_metrics[metric]), step)

    def get_learning_rate(learning_rate, hidden_dim, learning_rate_warmup_steps, step):
        learning_rate *= (hidden_dim ** -0.5)
        # Apply linear warmup
        learning_rate *= min(1.0, step / learning_rate_warmup_steps)
        # Apply rsqrt decay
        learning_rate *= (max(step, learning_rate_warmup_steps))**-0.5

        return learning_rate


    ################################ TRAINING STARTING####################################
    # def Do_Train(args, mode=''):
    if args.do_train:
        training_logs = []
        training_logs_mean = []
        if args.task == '1c':
            begin_pq_step = args.max_steps
        else:
            begin_pq_step = args.max_steps - args.stepsforpath

#        Training Loop
        total_trained_num_triples = 0
        # _ = evaluate_val(tasks)
        for step in range(init_step, args.max_steps+1):
            # if step == 3000:
            #     # print('setting regularization to 0')
            #     args.regularization = 0.0

            if '1c' in tasks:
                # print("1c")
                log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator, args, step)
                for metric in log:
                    writer.add_scalar('train/1c_'+metric, log[metric], step)
                training_logs.append(log)

                total_trained_num_triples += num_triples_per_step

                # if (step % args.log_steps == 0):
                #     logging.info('current learning_rate: %f' % (current_learning_rate))
                #     metrics = {}
                #     for metric in training_logs[0].keys():
                #         metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                #     log_metrics('Training average 1c', step, metrics)
                #     training_logs = []

            # ## decide which temporal task will be trained
            # task_selector = np.random.rand(1)
            # task_index = np.nonzero(task_selector < ratio_temporal_triples)[0][0]
            # selected_task = temporal_tasks[task_index]

            ## combine the datasets of 2i and 3i-2i
            # if '2i' in tasks:
            #     # start = time.time()
            #     log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator_2i, args, step, use_time=True)
            #     # end = time.time()
            #     # print('time used in training in total in 2i', end - start)

            #     for metric in log:
            #         writer.add_scalar('train/2i_'+metric, log[metric], step)
            #     training_logs.append(log)

            #     total_trained_num_triples += num_triples_per_step
            

                # if (step % args.log_steps == 0):
                #     logging.info('current learning_rate: %f' % (current_learning_rate))
                #     metrics = {}
                #     for metric in training_logs[0].keys():
                #         if metric == 'inter_loss':
                #             continue
                #         metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                #     if step % args.valid_steps == 0 and step > 0 and args.evaluate_train:
                #         evaluate_train()
                #     inter_loss_sum = 0.
                #     inter_loss_num = 0.
                #     for log in training_logs:
                #         if 'inter_loss' in log:
                #             inter_loss_sum += log['inter_loss']
                #             inter_loss_num += 1
                #     if inter_loss_num != 0:
                #         metrics['inter_loss'] = inter_loss_sum / inter_loss_num
                #     log_metrics('Training average 2i', step, metrics)
                #     training_logs = []

            # if '3i-2i' in tasks:
            #     start = time.time()
            #     log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator_3i_2i, args, step, use_time=True)
            #     end = time.time()
            #     print('time used in training in total in 3i-2i', end - start)

            #     for metric in log:
            #         writer.add_scalar('train/3i_2i_'+metric, log[metric], step)
            #     training_logs.append(log)

            #     total_trained_num_triples += num_triples_per_step

            #     if (step % args.log_steps == 0):
            #         logging.info('current learning_rate: %f' % (current_learning_rate))
            #         metrics = {}
            #         for metric in training_logs[0].keys():
            #             if metric == 'inter_loss':
            #                 continue
            #             metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

            #         if step % args.valid_steps == 0 and step > 0 and args.evaluate_train:
            #             evaluate_train()
            #         inter_loss_sum = 0.
            #         inter_loss_num = 0.
            #         for log in training_logs:
            #             if 'inter_loss' in log:
            #                 inter_loss_sum += log['inter_loss']
            #                 inter_loss_num += 1
            #         if inter_loss_num != 0:
            #             metrics['inter_loss'] = inter_loss_sum / inter_loss_num
            #         log_metrics('Training average 3i-2i', step, metrics)
            #         training_logs = []


             # also contain three parts: train_iterator_3i, train_iterator_3i_begin, train_iterator_3i_end
                # start = time.time()
            if len(args.negative_sample_types)==2:
                if '3i' in tasks:
                    # if step %2 == 0:
                    log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator_3i, args, step, use_time=True)
                    # else:
                    #     log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator_3i_time, args, step, use_time=True)

                    for metric in log:
                        writer.add_scalar('train/3i_'+metric, log[metric], step)
                    training_logs.append(log)

                    total_trained_num_triples += num_triples_per_step

                # print('next...')
                if '3i-2i' in tasks:
                    # if step%2 != 0:
                    log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator_3i_2i, args, step, use_time=True)
                    # else:
                    #     log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator_3i_2i_time, args, step, use_time=True)
                    # end = time.time()
                    # print('time used in training in total in 2i', end - start)

                    for metric in log:
                        writer.add_scalar('train/3i_2i_'+metric, log[metric], step)
                    training_logs.append(log)

                    total_trained_num_triples += num_triples_per_step
                    # end = time.time()
                    # print('time used in training in total in 3i', end - start)
            elif 'tail-batch' in args.negative_sample_types:
                if '3i' in tasks:
                    log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator_3i, args, step, use_time=True)

                    for metric in log:
                            writer.add_scalar('train/3i_'+metric, log[metric], step)
                    training_logs.append(log)

                    total_trained_num_triples += num_triples_per_step

                if '3i-2i' in tasks:
                    # start = time.time()
                    log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator_3i_2i, args, step, use_time=True)
                    # end = time.time()
                    # print('time used in training in total in 2i', end - start)

                    for metric in log:
                        writer.add_scalar('train/3i_2i_'+metric, log[metric], step)
                    training_logs.append(log)

                    total_trained_num_triples += num_triples_per_step

            elif 'time-batch' in args.negative_sample_types:
                if '3i' in tasks:
                    log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator_3i_time, args, step, use_time=True)

                    for metric in log:
                        writer.add_scalar('train/3i_'+metric, log[metric], step)
                    training_logs.append(log)

                    total_trained_num_triples += num_triples_per_step

                if '3i-2i' in tasks:
                    # start = time.time()
                    log, num_triples_per_step = query2box.train_step(query2box, optimizer, train_iterator_3i_2i_time, args, step, use_time=True)
                    # end = time.time()
                    # print('time used in training in total in 2i', end - start)

                    for metric in log:
                        writer.add_scalar('train/3i_2i_'+metric, log[metric], step)
                    training_logs.append(log)

                    total_trained_num_triples += num_triples_per_step
            else:
                raise NotImplementedError
            

            if (step % args.log_steps == 0):
                logging.info('current learning_rate: %f' % (current_learning_rate))
                metrics = {}
                for metric in training_logs[-1].keys():
                    # if metric == 'inter_loss':
                    #     continue
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                if step % args.valid_steps == 0 and step > 0 and args.evaluate_train:
                    evaluate_train()
                # inter_loss_sum = 0.
                # inter_loss_num = 0.
                # for log in training_logs:
                #     if 'inter_loss' in log:
                #         inter_loss_sum += log['inter_loss']
                #         inter_loss_num += 1
                # if inter_loss_num != 0:
                #     metrics['inter_loss'] = inter_loss_sum / inter_loss_num
                log_metrics('Training average loss', step, metrics)
                writer.add_scalar('train/average_loss', metrics['loss'], step)
                training_logs = []

            # if '2c' in tasks:
            #     log = query2box.train_step(query2box, optimizer, train_iterator_2, args, step)
            #     for metric in log:
            #         writer.add_scalar('2c_'+metric, log[metric], step)
            #     training_logs.append(log)
            
            # if '3c' in tasks:
            #     log = query2box.train_step(query2box, optimizer, train_iterator_3, args, step)
            #     for metric in log:
            #         writer.add_scalar('3c_'+metric, log[metric], step)
            #     training_logs.append(log)

            # writer.add_embedding(query2box.entity_embedding, metadata=meta)
            # writer.add_embedding(query2box.relation_embedding, metadata=meta)
            # writer.add_embedding(query2box.time_embedding, metadata=meta)

            # if training_logs == []:
            #     raise Exception("No tasks are trained!!")

            # if step >= warm_up_steps:
            #     logging.info('warm up step: %d' % (warm_up_steps))
            #     current_learning_rate = current_learning_rate / 10
            #     logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
            #     optimizer = torch.optim.Adam(
            #         filter(lambda p: p.requires_grad, query2box.parameters()), 
            #         lr=current_learning_rate
            #     )
            #     warm_up_steps = 2*warm_up_steps 

            
            # current_learning_rate = get_learning_rate(args.learning_rate, args.hidden_dim, warm_up_steps, step)
            # logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
            # optimizer = torch.optim.Adam(
            #     filter(lambda p: p.requires_grad, query2box.parameters()), 
            #     lr=current_learning_rate
            # )

            if step%args.model_save_step==0 and step > 0:
                save_variable_list = {
                    'curr_valid_mrrm':last_valid_mrrm,
                    'best_valid_mrrm':best_valid_mrrm,
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    # 'warm_up_steps': warm_up_steps
                }
                save_model(query2box, optimizer, save_variable_list, args)
            
            if args.do_valid and step % args.valid_steps == 0 and (step > 0):
                logging.info('Evaluating on Valid Dataset: %f epochs:', total_trained_num_triples/num_triples_per_epoch)
                last_valid_mrrm = evaluate_val(tasks)
                
                if last_valid_mrrm > best_valid_mrrm:
                    save_variable_list = {
                    'best_valid_mrrm':last_valid_mrrm,
                    'curr_valid_mrrm':last_valid_mrrm,
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    # 'warm_up_steps': warm_up_steps
                    }
                    save_model(query2box, optimizer, save_variable_list, args, best_model=True)
                    logging.info('Update checkpoints to a better valid mrr: %3.3f', last_valid_mrrm)
                    best_valid_mrrm = last_valid_mrrm

                # if step >= 3000:
                scheduler.step(best_valid_mrrm)
                # tune.report(iterations=step, valid_mrr=best_valid_mrrm)
                ## tune
                # if mode == 'TUNE':
                
           
        
    # try:`
    #     print(step)
    # except:
    #     step = 0

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        evaluate_val(tasks)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        evaluate_test()

    # if args.evaluate_train:
    #     logging.info('Evaluating on Training Dataset...')
    #     evaluate_train()

    print ('Training finished!!')
    logging.info("training finished!!")

    # Do_Train(args, mode)

if __name__ == '__main__':
    main(parse_args())