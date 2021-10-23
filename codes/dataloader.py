#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
from utils import *
TIME_TYPE = {'no-time':0, 'point-in-time':1, 'only-begin':2, 'only-end':3, 'full-interval':4}

'''
(1) It is meaningless to sample time points for static statements or statements with missing temporal information;
(2) For temporal statements, we sample time points that are not within the validity time period
    * for partial statements, we sample time from the previous part or the afterwards depending on the mention 'until ..., before'; 
    we need to ensure that it is still valid for some boxes but invalid for the intersection of the boxes; 
    * for full-interval statements, excluding them out of the sampled time points;
        * It is also unknown whether two kinds of negative samples are compementary or completing;
        * try to train these two negative samples separately; if it does not work, then probably try to do them separately. 
'''



class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, train_ans, mode):
        # assert mode == 'tail-batch'
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        # self.count = self.count_frequency(triples, train_ans, mode)
        self.true_missing = train_ans
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.mode == 'tail-batch':
            positive_sample = self.triples[idx][0] # the data format (sub, (rel, ), fake_obj)
            head, relations = positive_sample
            candicates = list(self.true_missing[positive_sample])
            # tail = np.random.choice(candicates)
            tail = self.triples[idx][1]
            subsampling_weight = len(candicates) + 4
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
            negative_sample_list = []
            negative_sample_size = 0
            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
                mask = np.in1d( ## mask is used here to filter out other true possibilities
                    negative_sample, 
                    self.true_missing[positive_sample], 
                    assume_unique=True, 
                    invert=True
                )
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)
            positive_sample = torch.LongTensor([positive_sample[0], positive_sample[1], tail])
            return positive_sample, negative_sample, subsampling_weight, self.mode
        elif self.mode == 'head-batch':
            positive_sample = self.triples[idx][0] # the data format (fake_sub, ((rel, ), obj))
            head, relations, tail = positive_sample 
            head = np.random.choice(list(self.true_missing[(0,(relations, tail))]))
            subsampling_weight = self.count[(relations, tail)] 
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
            negative_sample_list = []
            negative_sample_size = 0
            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_missing[(0,(relations, tail))], 
                    assume_unique=True, 
                    invert=True
                )
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)
            positive_sample = torch.LongTensor([head] + [i for i in relations] + [tail])
            return positive_sample, negative_sample, subsampling_weight, self.mode
            
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, None, subsample_weight, mode
    
    def get_subsampling_weight(new_query, true_missing, start=4):
        return len(true_missing[new_query]) + start

    @staticmethod
    def count_frequency(triples, true_missing, mode, start=4):
        count = {}
        # if mode == 'tail-batch':
        for triple in triples:
            # head, relations = triple
            if triple not in count:
                count[triple] = start + len(true_missing[triple])
        return count
        # elif mode == 'head-batch':
        #     for triple, qtype in triples:
        #         head, relations, tail = triple
        #         assert (relations, tail) not in count
        #         count[(relations, tail)] = start + len(true_missing[(0,(relations, tail))])
        #     return count
    
class TrainInterDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, ntimestamps, negative_sample_size, train_ans, mode, ans_t, use_one_sample=False, use_two_sample=False, add_hard_neg=False, double_point_in_time=False, num_time_negatives=0):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.ntimestamps = ntimestamps
        self.negative_sample_size = negative_sample_size - num_time_negatives
        self.mode = mode
        self.true_missing = train_ans
        self.true_ans_t = ans_t
        self.qtype = self.triples[0][-1]
        self.use_one_sample = use_one_sample
        self.use_two_sample = use_two_sample
        self.add_hard_neg = add_hard_neg
        self.num_time_negatives = num_time_negatives

        if double_point_in_time:
            self.qtype = '2-3-inter'

        # if self.qtype == '2-inter' or ((~self.use_one_sample) and (~self.use_two_sample) and self.qtype=='3-inter'):
        #     self.count = self.count_frequency(triples, train_ans)

        if self.use_one_sample:
            self.qtype = '2-inter'

        assert use_one_sample * use_two_sample != 1

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][0] #[entity, relation, start, end]
        gold_tail = self.triples[idx][1]
        time_type = self.triples[idx][2]

        # flat_query = np.array(query) # 
            ### deal with full interval 
        if time_type == TIME_TYPE['full-interval']:
            start_date = query[2]
            end_date = query[3]
            if self.use_one_sample: # sample one time from the inside
                assert start_date != end_date # we must make sure we do not consider point in time here, excluding tuples with point-in-time
                assert start_date != -1 # we must make sure we do not consider point in time here, excluding tuples with point-in-time
                assert end_date != -1 # we must make sure we do not consider point in time here, excluding tuples with point-in-time
                date = (np.random.random(1)*(end_date-start_date)+start_date).round()
                date = int(date[0])
                query = (query[0], query[1], date) 
            elif self.use_two_sample:
                assert start_date != end_date # we must make sure we do not consider point in time here, excluding tuples with point-in-time
                date_1,date_2 = (np.random.random(2)*(end_date-start_date)+start_date).round()
                while (date_1 == date_2):
                    date_2 = (np.random.random(1)*(end_date-start_date)+start_date).round()
                s_date, e_date = sorted([date_1, date_2])
                query = (query[0], query[1], int(s_date), int(e_date))
        # elif time_type == TIME_TYPE['only-begin']:
        #     query = (query[0], query[1], query[2]) 
        elif time_type in [TIME_TYPE['only-end'], TIME_TYPE['point-in-time'], TIME_TYPE['only-begin']]:
            query = query
        else:
            print("test ", self.triples[idx])
            raise NotImplementedError

        # tail = np.random.choice(list(self.true_missing[query]))
        positive_sample = torch.LongTensor(list(query)+[gold_tail]) # positive samples include the flat-query but the negative samples just have the tails    
        
        if 'tail-batch' in self.mode:
            subsampling_weight = self.get_subsampling_weight(query, self.true_missing)
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
            negative_sample_list = []
            negative_sample_size = 0
            ## in this part, for a query, hard negatives are those not in (s,r,o,t) but in (s, r, o)
            if self.add_hard_neg:
                assert len(self.true_missing[(query[0], query[1])]) >= len(self.true_missing[query])
                hard_neg = np.array(list(set(self.true_missing[(query[0], query[1])]) - set(self.true_missing[query])), dtype=np.int)
                # if hard_neg.size != 0:
                negative_sample_list.append(hard_neg)
                negative_sample_size += hard_neg.size
                #print('qtype:', self.qtype)
            ##
            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_missing[query], 
                    assume_unique=True, 
                    invert=True
                )
                negative_sample = negative_sample[mask]

                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
            
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)

            time_negative_sample = None 
            # return positive_sample, negative_sample, subsampling_weight, self.mode
            if 'time-batch' in self.mode: ## negative timestamps;
                negative_sample_list_time = []
                negative_sample_size = 0
                sro = (query[0], query[1], gold_tail)
                # subsampling_weight = self.get_subsampling_weight(sro, self.true_missing)
                # subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
                groundtruth_ts = self.true_ans_t[sro]
                if time_type == TIME_TYPE['only-begin']:
                    groundtruth_ts.update(set(np.arange(query[2], min(query[2]+20, self.ntimestamps), 1)) )       
                elif time_type == TIME_TYPE['only-end']:
                    groundtruth_ts.update(set(np.arange(max(0, query[2]-20), query[2], 1)))
                # elif time_type in [TIME_TYPE['point-in-time'], TIME_TYPE['full-interval']]:
                #     groundtruth_ts = self.true_ans_t[sro]
                # else:
                #     raise ValueError
                while negative_sample_size < self.num_time_negatives:
                    time_negative_sample = np.random.randint(self.ntimestamps, size=self.num_time_negatives*2)
                    mask = np.in1d(
                        time_negative_sample, 
                        groundtruth_ts, 
                        assume_unique=True, 
                        invert=True
                    )
                    time_negative_sample = time_negative_sample[mask]

                    negative_sample_list_time.append(time_negative_sample)
                    negative_sample_size += time_negative_sample.size
            
                time_negative_sample = np.concatenate(negative_sample_list_time)[:self.num_time_negatives]
                time_negative_sample = torch.from_numpy(time_negative_sample)

            return positive_sample, negative_sample, time_negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        if data[0][2] == None:
            time_negative_sample = None
        else:
            time_negative_sample = torch.stack([_[2] for _ in data], dim=0)
        subsample_weight = torch.cat([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return positive_sample, negative_sample, time_negative_sample, subsample_weight, mode
    
    @staticmethod
    def get_subsampling_weight(new_query, true_missing, start=4):
        return len(true_missing[new_query]) + start

    @staticmethod
    def count_frequency(triples, true_missing, start=4):
        count = {}
        for triple,qtype in triples:
            # query = triple[:-2]
            if triple not in count:
                count[query] = start + len(true_missing[triple])
        return count

class TestInterDataset(Dataset):
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, ntimestamps, mode, use_one_sample=False, use_two_sample=False, enumerate_time=False, double_point_in_time=False, predict_o=True, predict_t=False, predict_r=False):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.ntimestamps = ntimestamps
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]
        self.use_one_sample = use_one_sample 
        self.use_two_sample = use_two_sample
        self.enumerate_time = enumerate_time
        self.predict_o = predict_o
        self.predict_t = predict_t
        self.predict_r = predict_r

        if double_point_in_time:
            self.qtype = '2-3-inter'

        if self.use_one_sample: ## Once datasets are fed here, must be 2-inter or 3-inter
            self.qtype = '2-inter'
        if self.enumerate_time: ## Once datasets are fed here, must be 2-inter or 3-inter
            self.qtype = '2-inter'

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][0] # used as keys to search for the filtered set
        gold_tail = self.triples[idx][1]
        time_type = self.triples[idx][2]

        if self.predict_o:
        # tail = self.triples[idx][-2] # this is not the true tail, used for place holder
            negative_sample = torch.LongTensor(range(self.nentity)) # negative samples: all the entities
            # flat_query = np.array(query)

            # flat_query = np.array(query) # [entity, relation, start, end, time_interval]
            ### deal with full interval 
            if time_type == TIME_TYPE['full-interval']:
                start_date = query[2]
                end_date = query[3]
                if self.enumerate_time: 
                    ## enumerate all the timestamps between two points; need to use the interval length to calculate the metrics;
                    ## expand <s, r, o, b, e> to <s, r, o, b, e, interval_length>
                    num_ts = end_date - start_date + 1
                    ts = np.arange(start_date, end_date+1, 1)
                    s = np.repeat(query[0], num_ts,  axis=0)
                    r = np.repeat(query[1], num_ts,  axis=0)
                    # o = np.zeros_like(ts)
                    o = np.repeat(gold_tail, num_ts,  axis=0)
                    positive_sample = np.stack([s, r, ts, o]).transpose()
                    positive_sample = torch.LongTensor(positive_sample)
                    query = [tuple(i) for i in zip(s, r, ts)]
                    return positive_sample, negative_sample, self.mode, query
                elif self.use_one_sample: # sample one time from the inside; note that we change '3-inter' to '2-inter' when initializing.
                    assert self.qtype == '2-inter'
                    assert start_date != end_date # we must make sure we do not consider point in time here, excluding tuples with point-in-time
                    date = (np.random.random(1)*(end_date-start_date)+start_date).round()
                    date = int(date[0])
                    # flat_query = flat_query[:4]
                    # assert flat_query[0] == flat_query[2]
                    query = (query[0], query[1], date) 
                elif self.use_two_sample:
                    assert self.qtype == '3-inter'
                    assert start_date != end_date # we must make sure we do not consider point in time here, excluding tuples with point-in-time
                    date_1,date_2 = (np.random.random(2)*(end_date-start_date)+start_date).round()
                    while (date_1 == date_2):
                        date_2 = (np.random.random(1)*(end_date-start_date)+start_date).round()
                    s_date, e_date = sorted([date_1, date_2])
                    # flat_query[3] = s_date
                    # flat_query[5] = e_date
                    query = (query[0], query[1], int(s_date), int(e_date))            
                # query = ((flat_query[0], (flat_query[1],)), (flat_query[2], (flat_query[3],)), (flat_query[4], (flat_query[5],))) 
            # elif  time_type == TIME_TYPE['only-begin']: 
            #     query = (query[0], query[1], query[2]) 
            elif time_type in [TIME_TYPE['only-end'], TIME_TYPE['point-in-time'], TIME_TYPE['only-begin']]:
                query = query
            else:
                print(query)
                print(time_type)
                raise NotImplementedError

            positive_sample = torch.LongTensor(list(query)+[gold_tail])
            return positive_sample, negative_sample, self.mode, query
        if self.predict_t:
            negative_sample = torch.LongTensor(range(self.ntimestamps))
            if time_type == 'full-interval':
                query = (query[0], query[1], (query[2], query[3]))
            positive_sample =  torch.LongTensor(list(query)+[gold_tail]) ## all the triples are represented in <s, r, t, o>; when it is full interval, <s, r, (b, e), o>
            return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query

class TestChainInterDataset(Dataset):
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][:-2]
        tail = self.triples[idx][-2]
        negative_sample = torch.LongTensor(range(self.nentity))
        positive_sample = torch.LongTensor([query[0][0], query[0][1][0], query[0][1][1], query[1][0], query[1][1][0]]+[self.triples[idx][-2]])
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query

class TestInterChainDataset(Dataset):
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][:-2]
        tail = self.triples[idx][-2]
        negative_sample = torch.LongTensor(range(self.nentity))
        positive_sample = torch.LongTensor([query[0][0], query[0][1][0], query[1][0], query[1][1][0], query[2]]+[self.triples[idx][-2]])
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query

class TestDataset(Dataset):
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][0]
        gold_tail = self.triples[idx][1]
        negative_sample = torch.LongTensor(range(self.nentity))
        positive_sample = torch.LongTensor(list(query) + [gold_tail])
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader_tail, qtype):
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.qtype = qtype
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data