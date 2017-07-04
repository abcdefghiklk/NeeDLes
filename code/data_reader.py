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

'''
Loading all the experiment data from file stream
'''
def load_data(bug_content_path, code_content_path, file_oracle_path, sequence_oracle_path, split_length = -1, encoding = 'gbk'):

    #Read code contents
    code_contents = load_code_contents(code_content_path, split_length = split_length, encoding = encoding)

    #Read bug contents
    bug_contents = load_bug_contents(bug_content_path, encoding = encoding)

    #Read true relevant code sequences for each bug
    sequence_oracle = load_code_contents(sequence_oracle_path, encoding = encoding)

    #Read true relevant files for each bug
    file_oracle = read_oracle(file_oracle_path)

    return (bug_contents,code_contents, file_oracle, sequence_oracle)


'''
Loading all the bug contents
File format: one bug sequence per line
'''
def load_bug_contents(file_path, encoding = 'gbk'):
    data_input = codecs.open(file_path, encoding = encoding)
    lines = data_input.readlines()
    data_input.close()
    content_list = []
    for line in lines:
        if(len(line.strip())>0):
            content_list.append(line.strip())
    return content_list


'''
Loading all the code contents
File format: Each line represents sequence for one code file.
Each line is splitted into methods by "\t".
Each method is splitted into sequences according to the given length
'''
def load_code_contents(file_path, split_length = -1, encoding = 'gbk'):
    data_input = codecs.open(file_path, encoding = encoding)
    lines = data_input.readlines()
    data_input.close()
    content_list = []
    #Traversing each line
    for line in lines:
        # if line is null, then append void data
        if len(line.strip())<=2:
            content_list.append([])
        else:
            # Split into methods
            method_list = line.strip().split('\t')

            # If we do not further split methods into chunks
            # Directly adding the method list to the final list
            if split_length == -1:
                content_list.append(method_list)

            # Else we traverse each method
            else:
                chunk_list = []
                for one_method in method_list:
                    # Get the word sequence for the method
                    terms = one_method.split(' ')

                    # Compute the number of chunks for this method
                    chunk_num = int(math.ceil(len(terms)/split_length))

                    # Determine the start and end for each chunk
                    for i in range(chunk_num):
                        start = i*split_length
                        end = (i+1)*split_length
                        if end > len(terms):
                            end = len(terms)

                        chunk_list.append(" ".join(terms[start:end]))

                # Appending the method sequence list to the final list
                content_list.append(chunk_list)

    return content_list

'''
Reading the true relevant files for each bug from file stream
Format: each line: bugID\tfileID\t(1/0)
'''
def read_oracle(file_oracle_path):
    data_input = codecs.open(file_oracle_path)
    lines = data_input.readlines()
    data_input.close()
    oracle_per_bug = []
    positive_index_list = []
    negative_index_list = []
    bug_index = 0

    #Traversing each line
    for line in lines:

        string_list = line.split('\t')
        if(len(string_list)==3):
            #Still the previous bug ID
            if(int(string_list[0]) == bug_index):
                #If relevant, appending to the positive list
                if(int(string_list[2])==1):
                    positive_index_list.append(int(string_list[1]))

                #Else, appending to the negative list
                else:
                    negative_index_list.append(int(string_list[1]))

            #A new bug ID
            else:
                #Append the previous bug oracle
                oracle_per_bug.append([positive_index_list,negative_index_list])
                positive_index_list = []
                negative_index_list = []

                #Set the new bug index
                bug_index = int(string_list[0])

                #Processing this line
                if(int(string_list[2])==1):
                    positive_index_list.append(int(string_list[1]))
                else:
                    negative_index_list.append(int(string_list[1]))

    #Appending the last bug oracle
    oracle_per_bug.append([positive_index_list,negative_index_list])
    return oracle_per_bug





