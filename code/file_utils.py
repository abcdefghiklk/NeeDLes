# -*- coding:utf8 -*-
import os
import sys
import codecs
import numpy as np
import argparse
import time
import re
import math

def form(value):
    return "%.3f" % value

def load_data(bug_content_path, code_content_path, file_oracle_path, method_oracle_path, split_length = -1, encoding = 'gbk'):
    code_contents = load_code_contents(code_content_path, split_length = split_length, encoding = encoding)
    bug_contents = load_bug_contents(bug_content_path, encoding = encoding)
    method_oracle = load_code_contents(method_oracle_path, encoding = encoding)
    file_oracle = read_oracle(file_oracle_path)

    return(bug_contents,code_contents, file_oracle, method_oracle)


def load_bug_contents(file_path, encoding = 'gbk'):
    data_input = codecs.open(file_path, encoding = encoding)
    lines = data_input.readlines()
    data_input.close()
    content_list = []
    for line in lines:
        if(len(line.strip())>0):
            content_list.append(line.strip())
    return content_list

def load_code_contents(file_path, split_length = -1, encoding = 'gbk'):
    data_input = codecs.open(file_path, encoding = encoding)
    lines = data_input.readlines()
    data_input.close()
    content_list = []
    for line in lines:
        if len(line.strip())<=2:
            content_list.append([])
        else:
            method_list = line.strip().split('\t')
            if split_length == -1:
                content_list.append(method_list)
            else:
                chunk_list = []
                for one_method in method_list:
                    terms = one_method.split(' ')
                    chunk_num = int(math.ceil(len(terms)/split_length))
                    for i in range(chunk_num):
                        start = i*split_length
                        end = (i+1)*split_length
                        if end > len(terms):
                            end = len(terms)
                        chunk_list.append(one_method[start:end])
                content_list.append(chunk_list)

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





