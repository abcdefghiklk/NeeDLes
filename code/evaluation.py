import os
import sys
import codecs
import numpy as np
import argparse
import time
import re
import math

def evaluate(prediction_list,positive_index_list,k = 10,rel_threshold = 0.65):
    evaluation_result_list = []
    for i in range(len(prediction_list)):
        evaluation_result_list.append(evaluate_one_bug(prediction_list[i], positive_index_list[i], k, rel_threshold))
    return evaluation_result_list

def evaluate_one_bug(predictions,positive_index,k = 10,rel_threshold = 0.65):
    true_positive = 0
    positive_num = 0

    #precision, recall based
    for i in range(len(predictions)):
        if(predictions[i]>rel_threshold):
            positive_num+=1
            if(i in positive_index):
                true_positive+=1
    if(positive_num==0):
        precision = 0
    else:
        precision = true_positive/positive_num
    recall = true_positive/len(positive_index)

    #rank based
    values = [j for j in predictions]
    ranks = [1] * len(positive_index)
    for s in range(len(positive_index)):
        one_index = positive_index[s]
        this_value =  values[one_index]
        for i in range(len(values)):
            if(values[i]>this_value):
                ranks[s]+=1

    #average_precision
    ordered_ranks = sorted(ranks)
    average_precision = 0
    for k in range(len(ordered_ranks)):
        average_precision += (k+1)/ordered_ranks[k]
    average_precision=average_precision/len(ordered_ranks)

    #mrr
    mrr = 1/ordered_ranks[0]

    #topk
    if(ordered_ranks[-1]<=k):
        topk = 1
    else:
        topk = 0
    return average_precision,mrr,topk


def export_evaluation(evaluations, evaluation_file_path):
    data_output = codecs.open(evaluation_file_path,'w')
    for one_evaluation in evaluations:
        evaluation_string = "avg_precision = {}\t mrr = {}\ttopk = {}\n".format(one_evaluation[0],one_evaluation[1],one_evaluation[2])
        data_output.write(evaluation_string)
    data_output.close()

def export_one_evaluation(one_evaluation, evaluation_file_path):
    data_output = codecs.open(evaluation_file_path,'a+')
    evaluation_string = "avg_precision = {}\t mrr = {}\ttopk = {}\n".format(one_evaluation[0],one_evaluation[1],one_evaluation[2])
    data_output.write(evaluation_string)
    data_output.close()
