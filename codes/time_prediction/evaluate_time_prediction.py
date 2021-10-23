import sys
sys.path.append('../')

import pickle
import torch
import argparse
import numpy
from collections import defaultdict
import pandas as pd

from time_prediction.evaluate_helper import compute_scores, compute_preds, \
    stack_tensor_list, get_thresholds, load_pickle, prepare_data_iou_scores

from time_prediction.interval_metrics import smooth_iou_score, gaeiou_score, aeiou_score, tac_score, giou_score, precision_score, \
    recall_score

## code is from the tkbi github.
time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5, 'time_type':6}


def compute_interval_scores(valid_time_scores_info, test_time_scores_info, id_year_map, save_time_results=None,
                            method='greedy-coalescing'):
    """
    test_time_scores_info: a list of results: [(s, r, o, start, end), t_scores, years]
    valid_time_scores_info: a list of results: [(s, r, o, start, end), t_scores, years]
    id_year_map: a dictionary to convert id to year
    Using these time scores, depending on method it predicts intervals for each fact in test kb (a ranking ideally instead of a single interval),
    and returns gIOU/aeIOU/IOU scores @1/5/10
    """
    thresholds = None

    print("Using method {}".format(method))

    if method == 'greedy-coalescing':
        aggr = 'mean'
        thresholds = get_thresholds(valid_time_scores_info, test_time_scores_info, aggr=aggr)
        print("Computed thresholds (aggr= {})\n".format(aggr))
    else:
        raise NotImplementedError

    # elif method in ['start-end-exhaustive-sweep']:  # for time boundary models
    #     start_t_scores, end_t_scores = t_scores

    #     start_t_scores = stack_tensor_list(start_t_scores)
    #     end_t_scores = stack_tensor_list(end_t_scores)

    #     t_scores = (start_t_scores, end_t_scores)

    # load valid scores

    # compute thresholds

    # id_year_map = id_year_map.long()

    # id_year_map_dict = {}
    # for i, j in enumerate(id_year_map):
    #     id_year_map_dict[i] = j
    # print(id_year_map_dict)
    # for i, j in zip(t_gold_min[:5], t_gold_max[:5]):
    #     print("gold start:{}, gold end:{}".format(i, j))

    # ----------------------#
    print("**************")
    topk_ranks = 10

    # score_func = {"precision":precision_score, "recall":recall_score, "aeIOU": aeiou_score, "TAC": tac_score, "IOU": smooth_iou_score, "gIOU": giou_score}
    score_func = {"gIOU": giou_score, "aeIOU": aeiou_score, "TAC": tac_score, "gaeIOU":gaeiou_score}

    scores_dict = {}  # for saving later

    scores_dict_facts = {}

    for score_to_compute in score_func.keys():
        print("\nScore:{}".format(score_to_compute))
        # iou_scores= compute_scores(ktrain, facts, t_scores, method=method, durations=durations,
        # 			  score_func=score_func[score_to_compute], use_time_interval=use_time_interval, topk_ranks=topk_ranks)
        iou_scores, t_gold_min, t_gold_max, t_pred_min, t_pred_max = compute_scores(id_year_map, test_time_scores_info, method=method,
                                    thresholds=thresholds,
                                    score_func=score_func[score_to_compute], topk_ranks=topk_ranks)
        # output best iou @ k
        iouatk = [1, 5, 10]

        for i in iouatk:
            all_scores = torch.stack(iou_scores[:i])
            # print("all_scores shape:",all_scores.shape)
            best_scores, _ = torch.max(all_scores, 0)

            scores_dict[(i, score_to_compute)] = best_scores

            # print("best_scores shape:",best_scores.shape)
            print("Best {} @{}: {}".format(score_to_compute, i, torch.mean(best_scores)))
    
            ## by durtation
            scores_dict_facts[(i, score_to_compute)] = best_scores.cpu()

    if save_time_results is not None:
        # saves metrics of the form
        pickle_filename = "{}_time_scores_analysis".format(save_time_results)

        gold_min, gold_max, pred_min, pred_max = compute_preds(id_year_map, test_time_scores_info,
                                                               method=method, thresholds=thresholds,
                                                               score_func=score_func['gaeIOU'],
                                                               topk_ranks=topk_ranks)
        ## predict result for each query
        results = {}
        lines = []
        for i, info in enumerate(test_time_scores_info):
            results[info[0]] = [(gold_min[i], gold_max[i]), (pred_min[i], pred_max[i])]
            line = [info[0][0], info[0][1], info[0][2], gold_max[i].item()-gold_min[i].item()+1, scores_dict_facts[(1, 'gIOU')][i].item(), scores_dict_facts[(5, 'gIOU')][i].item(), scores_dict_facts[(10, 'gIOU')][i].item(),\
            scores_dict_facts[(1, 'aeIOU')][i].item(), scores_dict_facts[(5, 'aeIOU')][i].item(), scores_dict_facts[(10, 'aeIOU')][i].item(),\
            scores_dict_facts[(1, 'gaeIOU')][i].item(), scores_dict_facts[(5, 'gaeIOU')][i].item(), scores_dict_facts[(10, 'gaeIOU')][i].item()]
            lines.append(line)

        with open(pickle_filename, 'wb') as handle:
            pickle.dump(results, handle)
            print("\nPickled query, t_gold_min, t_gold_max, t_pred_min, t_pred_max to {}\n".format(pickle_filename))

        with open(pickle_filename+'_by_duration', 'wb') as handle:
            pickle.dump(lines, handle)

        with open(pickle_filename+'_scores', 'wb') as handle:
            pickle.dump(scores_dict, handle)
            # print("\nPickled query, t_gold_min, t_gold_max, t_pred_min, t_pred_max to {}\n".format(pickle_filename))

    else:
        print("\nNot saving scores")


# def get_time_scores(scoring_function, test_kb, method='greedy-coalescing', load_to_gpu=True):
#     """
#     Returns dict containing time scores for each fact in test_kb (along with some other useful stuff needed later)
#     For time-point models, this means scores for each possible time point (t scores for each fact).
#     For time-boundary models (not implemented yet), this would mean t start scores and t end scores for each fact.
#     """
#     facts = test_kb.facts

#     if method in ['greedy-coalescing']:  # for time-point models

#         scores_t_list = []

#         for i in range(0, int(facts.shape[0]), 1):
#             fact = facts[i]

#             s, r, o = fact[:3]

#             start_bin = fact[3 + time_index["t_s_orig"]]

#             # start_bin, end_bin=fact[3:5]

#             # num_times=end_bin-start_bin+1
#             num_times = 2

#             if num_times > 1:
#                 t = numpy.arange(start_bin, start_bin + 2)

#                 # t=numpy.arange(start_bin, end_bin+1)
#             else:
#                 num_times += 1
#                 # to avoid batch size of 1
#                 t = numpy.array([start_bin, start_bin])

#             s = numpy.repeat(s, num_times)
#             r = numpy.repeat(r, num_times)
#             o = numpy.repeat(o, num_times)

#             # '''

#             if load_to_gpu:
#                 s = torch.autograd.Variable(torch.from_numpy(
#                     s).cuda().unsqueeze(1), requires_grad=False)
#                 r = torch.autograd.Variable(torch.from_numpy(
#                     r).cuda().unsqueeze(1), requires_grad=False)
#                 o = torch.autograd.Variable(torch.from_numpy(
#                     o).cuda().unsqueeze(1), requires_grad=False)
#                 t = torch.autograd.Variable(torch.from_numpy(
#                     t).cuda().unsqueeze(1), requires_grad=False)
#             else:
#                 # CPU
#                 s = torch.autograd.Variable(torch.from_numpy(
#                     s).unsqueeze(1), requires_grad=False)
#                 r = torch.autograd.Variable(torch.from_numpy(
#                     r).unsqueeze(1), requires_grad=False)
#                 o = torch.autograd.Variable(torch.from_numpy(
#                     o).unsqueeze(1), requires_grad=False)
#                 t = torch.autograd.Variable(torch.from_numpy(
#                     t).unsqueeze(1), requires_grad=False)

#             # print(facts[i],facts_track_range, i,s.shape, facts_time_chunk, len(numpy.nonzero(facts_track_range==i)))

#             scores_t = scoring_function(s, r, o, None).data

#             # save for later (all scores_t are same pick any one)
#             # print('scores_t shape', scores_t.shape)
#             scores_t_list.append(scores_t[-1])

#         # print('the shape of scores_t', scores_t_list[0].shape)
#         # scores_t_pickle=torch.tensor(scores_t_pickle)
#         t = torch.from_numpy(facts[:, 3:]).unsqueeze(1)

#         data_pickle = prepare_data_iou_scores(
#             t, test_kb, scores_t=scores_t_list, load_to_gpu=load_to_gpu)
#         data_pickle["facts"] = facts
#         data_pickle["data_folder_full_path"] = test_kb.datamap.dataset_root

#     elif method in ["start-end-exhaustive-sweep"]:
#         num_relations = len(test_kb.datamap.relation_map)
#         start_scores_t_list = []
#         end_scores_t_list = []

#         for i in range(0, int(facts.shape[0]), 1):
#             fact = facts[i]
#             s, r, o = fact[:3]

#             s = numpy.repeat(s, 2)  # to avoid batch size of 1
#             r = numpy.repeat(r, 2)
#             o = numpy.repeat(o, 2)

#             if load_to_gpu:
#                 s = torch.autograd.Variable(torch.from_numpy(
#                     s).cuda().unsqueeze(1), requires_grad=False)
#                 r = torch.autograd.Variable(torch.from_numpy(
#                     r).cuda().unsqueeze(1), requires_grad=False)
#                 o = torch.autograd.Variable(torch.from_numpy(
#                     o).cuda().unsqueeze(1), requires_grad=False)
#             else:  # CPU
#                 s = torch.autograd.Variable(torch.from_numpy(
#                     s).unsqueeze(1), requires_grad=False)
#                 r = torch.autograd.Variable(torch.from_numpy(
#                     r).unsqueeze(1), requires_grad=False)
#                 o = torch.autograd.Variable(torch.from_numpy(
#                     o).unsqueeze(1), requires_grad=False)

#             start_scores_t = scoring_function(s, r, o, None).data
#             end_scores_t = scoring_function(s, r + num_relations, o, None).data

#             # save for later (all scores_t are same pick any one)
#             start_scores_t_list.append(start_scores_t[-1])
#             end_scores_t_list.append(end_scores_t[-1])

#         t = torch.from_numpy(facts[:, 3:]).unsqueeze(1)

#         data_pickle = prepare_data_iou_scores(
#             t, test_kb, scores_t=(start_scores_t_list, end_scores_t_list), load_to_gpu=load_to_gpu)
#         data_pickle["facts"] = facts
#         data_pickle["data_folder_full_path"] = test_kb.datamap.dataset_root

#     else:
#         raise Exception("Not implemented")

#     return data_pickle

# def read_data():


if __name__ == '__main__':
    ## read the scores from the disk
    # dataset = 'WIKIDATA12k_new'
    # dirpath = '/home/ling/Dynamic-Graph/query2box/logs/box/%s/BoxTransE/test_use_one_sample-reduce-sample-128-3000-lr0.003-noptr/rank_result' % dataset
    # timestamp_filepath = '/home/ling/Dynamic-Graph/query2box/data/%s' % dataset

    # ## read id_year map
    # print('dirpath', dirpath)
    # id_year_map = pd.read_csv(timestamp_filepath+'/timestamps.csv', header=0)
    # id_year_map = dict(zip(id_year_map['index'], id_year_map['year']))

    # ##read rank data; now we only consider 
    # test_3i_2i_test_info = pickle.load(open(dirpath+'/3i-2i_rank_time.pkl', 'rb'))
    # test_3i_info = pickle.load(open(dirpath+'/3i_rank_time.pkl', 'rb'))
    # # 2i_end_test_info = pickle.load(open(dirpath+'/2i-end_rank_time.pkl', 'rb'))
    # # 2i_begin_test_info = pickle.load(open(dirpath+'/2i-begin_rank_time.pkl', 'rb'))

    # valid_3i_2i_info = pickle.load(open(dirpath+'/3i-2i-valid_rank_time.pkl', 'rb'))
    # valid_3i_info = pickle.load(open(dirpath+'/3i-valid_rank_time.pkl', 'rb'))
    # # 2i_end_test_info = pickle.load(open(dirpath+'/2i-end_rank_time.pkl', 'rb'))
    # # 2i_begin_test_info = pickle.load(open(dirpath+'/2i-begin_rank_time.pkl', 'rb'))

    # valid_info = valid_3i_2i_info + valid_3i_info
    # test_info = test_3i_2i_test_info + test_3i_info

    # # print('the shape of valid info', valid_info[0])
    # # print('the shape of test info', len(test_info))

    # compute_interval_scores(valid_info, test_info, id_year_map, save_time_results=dirpath+'/time',
    #                         method='greedy-coalescing')


    # ## another datasetss
    dataset = 'wikidata_toy_new'
    dirpath = '/home/ling/Dynamic-Graph/query2box/logs/box/%s/BoxTransE/test-result_d400-128-neg-16/rank_result' % dataset
    timestamp_filepath = '/home/ling/Dynamic-Graph/query2box/data/%s' % 'wikidata_toy_new'

    ## read id_year map
    print('dirpath', dirpath)
    id_year_map = pd.read_csv(timestamp_filepath+'/timestamps.csv', header=0)
    id_year_map = dict(zip(id_year_map['index'], id_year_map['year']))

    ##read rank data; now we only consider 
    test_3i_2i_test_info = pickle.load(open(dirpath+'/3i-2i_rank_time.pkl', 'rb'))
    test_3i_info = pickle.load(open(dirpath+'/3i_rank_time.pkl', 'rb'))
    # 2i_end_test_info = pickle.load(open(dirpath+'/2i-end_rank_time.pkl', 'rb'))
    # 2i_begin_test_info = pickle.load(open(dirpath+'/2i-begin_rank_time.pkl', 'rb'))

    valid_3i_2i_info = pickle.load(open(dirpath+'/3i-2i-valid_rank_time.pkl', 'rb'))
    valid_3i_info = pickle.load(open(dirpath+'/3i-valid_rank_time.pkl', 'rb'))
    # 2i_end_test_info = pickle.load(open(dirpath+'/2i-end_rank_time.pkl', 'rb'))
    # 2i_begin_test_info = pickle.load(open(dirpath+'/2i-begin_rank_time.pkl', 'rb'))

    valid_info = valid_3i_2i_info + valid_3i_info
    test_info = test_3i_2i_test_info + test_3i_info

    print('the shape of valid info', valid_info[0])
    print('the shape of test info', len(test_info))

    compute_interval_scores(valid_info, test_info, id_year_map, save_time_results=dirpath+'/time',
                            method='greedy-coalescing')










