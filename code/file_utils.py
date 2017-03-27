import os
import sys
import codecs
import numpy as np
import argparse
import time
import re

def form(value):
    return "%.3f" % value

def export_predictions(oracle_list, prediction_list, dir_path):
    for i in range(len(oracle_list)):
        file_path = os.path.join(dir_path, "prediction_{}".format(i))
        data_out = codecs.open(file_path,'w')
        data_out.write(str(oracle_list[i]))
        data_out.write("\n")
        prediction_str = [form(a) for a in prediction_list[i]]
        data_out.write(str(prediction_str))
        data_out.close()

def load_data(bug_content_path, code_content_path, file_oracle_path, method_oracle_path):
    code_contents = load_contents(code_content_path)
    bug_contents = load_contents(bug_content_path)
    method_oracle = load_relevant_methods(method_oracle_path)
    file_oracle = read_oracle(file_oracle_path)

    return(bug_contents,code_contents, file_oracle, method_oracle)


def load_contents(file_path):
    data_input = codecs.open(file_path)
    lines = data_input.readlines()
    data_input.close()
    content_list = []
    for line in lines:
        if(len(line)>1):
            content_list.append(line)
    return content_list

def load_relevant_methods(file_path):
    data_input = codecs.open(file_path)
    lines = data_input.readlines()
    data_input.close()
    content_list = []
    for line in lines:
        content_list.append(line)
    return content_list

def read_oracle(oracle_file_path):
    data_input = codecs.open(oracle_file_path)
    lines = data_input.readlines()
    data_input.close()
    oracle_per_bug = []
    positive_index_list = []
    negative_index_list = []
    bug_index = 0
    for line in lines:
        string_list = line.split('\t')
        if(len(string_list)==3):
            if(int(string_list[0])==bug_index):
                if(int(string_list[2])==1):
                    positive_index_list.append(int(string_list[1]))
                else:
                    negative_index_list.append(int(string_list[1]))
            else:
                oracle_per_bug.append([positive_index_list,negative_index_list])
                positive_index_list = []
                negative_index_list = []
                bug_index = int(string_list[0])
                if(int(string_list[2])==1):
                    positive_index_list.append(int(string_list[1]))
                else:
                    negative_index_list.append(int(string_list[1]))
    oracle_per_bug.append([positive_index_list,negative_index_list])
    return oracle_per_bug

def export_evaluation(evaluations, evaluation_file_path):
    data_output = codecs.open(evaluation_file_path,'w')
    for one_evaluation in evaluations:
        evaluation_string = "precision = {}\trecall = {}\tavg_precision = {}\t mrr = {}\ttopk = {}\n".format(one_evaluation[0],one_evaluation[1],one_evaluation[2],one_evaluation[3],one_evaluation[4])
        data_output.write(evaluation_string)
    data_output.close()



