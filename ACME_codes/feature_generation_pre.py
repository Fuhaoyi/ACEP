import pandas as pd
import numpy as np
#--------------------------------------------------------

import os
import math


def getpssm(filename):
    f = open(filename, 'r')
    f.readline()
    f.readline()
    f.readline()
    linelist = f.readlines()
    pssm = []
    for i in linelist:
        if i!='\n':
            oneline = i.split()
            m = []
            m.append(oneline[1])
            #m.extend([int(i) for i in oneline[22:42]])   #--------------------------
            m.extend([int(i) for i in oneline[2:22]])
            pssm.append(m)
        else:
            break
    f.close()

    pssmdf = pd.DataFrame(pssm,columns=['seq','A','R','N','D','C','Q','E','G','H','I',
                                        'L','K','M','F','P','S','T','W','Y','V'])
    col = pssmdf.columns.insert(1, 'X')
    pssmdf = pssmdf.reindex(columns=col, fill_value=0)

    return pssmdf


def getpssmlist(dir):
    filenamelist = os.listdir(dir)
    pssmlist = []
    pssm_file_list = []
    for i in filenamelist:
        filename = os.path.join(dir,i)
        pssmlist.append(getpssm(filename))
        pssm_file_list.append(i)
    print('pssmlist is ok')
    print(len(pssm_file_list))
    #print(pssm_file_list)
    return  pssmlist, pssm_file_list




def psssmfeature(sequence_index,one_sequence, pssm_matrix_list):

    str1 = one_sequence
    padding_vst_pssm = -1
    sequence_pssm = []
    str2 = ''.join(pssm_matrix_list[sequence_index].iloc[:, 0].values)
    if str1 == str2:
        pssm = pssm_matrix_list[sequence_index].iloc[:, 1:22].values
        padding = np.zeros((200 - pssm.shape[0], 21), dtype=np.float32)
        padding_vst_pssm = np.vstack([padding, pssm])
        sequence_pssm = [sequence_index, pssm_matrix_list[sequence_index]]

    if type(padding_vst_pssm) == type(-1):
        print('error pssm in empty')
        print(sequence_index)
        print(one_sequence)

    sequence_index_out = sequence_index
    return sequence_index_out, padding_vst_pssm, sequence_pssm   #numpy  shape:(None,200,21)  (None,)



def encode2numerical(sequence_index, one_sequence):
    dict1 = {'X':0,
             'A':1,
             'R':2,
             'N':3,
             'D':4,
             'C':5,
             'Q':6,
             'E':7,
             'G':8,
             'H':9,
             'I':10,
             'L':11,
             'K':12,
             'M':13,
             'F':14,
             'P':15,
             'S':16,
             'T':17,
             'W':18,
             'Y':19,
             'V':20
             }

    str1 = one_sequence
    str1 = 'X' * (200 - len(str1)) + str1
    str2 = ''
    for j in str1:
        if j not in dict1.keys():
            j = 'X'
        str2 =str2 + str(dict1[j]) + ' '

    sequence_num = str2.split()
    sequence_num = [int(k) for k in sequence_num]
    sequence_index_out = sequence_index

    return sequence_index_out ,sequence_num



def one_hot_vector(sequence_index, one_sequence):
    sequence_index_out, sequence_num = encode2numerical(sequence_index, one_sequence)
    one_hot_matrix = np.zeros((200, 21), dtype=np.float32)
    for i in range(200):
        if sequence_num[i]!=0:
            one_hot_matrix[i,sequence_num[i]] = 1.0

    #return sequence_index_out, one_hot_matrix, sequence_label   #numpy   shape:(None,200,21) (None,)
    return one_hot_matrix   #numpy   shape:(None,200,21) (None,)

def AAC_ft(sequence_index_x, one_sequence_x):
    dict1 = {'X': 0,
             'A': 1,
             'R': 2,
             'N': 3,
             'D': 4,
             'C': 5,
             'Q': 6,
             'E': 7,
             'G': 8,
             'H': 9,
             'I': 10,
             'L': 11,
             'K': 12,
             'M': 13,
             'F': 14,
             'P': 15,
             'S': 16,
             'T': 17,
             'W': 18,
             'Y': 19,
             'V': 20
             }

    str1 = one_sequence_x
    str2 = ''
    for j in str1:
        if j not in dict1.keys():
            j = 'X'
        str2 = str2 + str(dict1[j]) + ' '

    sequence_num = str2.split()
    sequence_num = [int(k) for k in sequence_num]
    sequence_aac = np.zeros((21,),dtype=np.float32)
    for i in range(21):
        count = 0
        for j in sequence_num:
            if i == j:
                count = count+1
        sequence_aac[i] = count
    #sequence_aac = sequence_aac/len(sequence_num)
    sequence_index_out = sequence_index_x

    #return sequence_index_out, sequence_aac, sequence_label
    return sequence_aac


def ConvertSequence2Feature(sequence_data, pssmdir):
    sequence_count = sequence_data.shape[0]
    print(sequence_data)
    feature_pssm = np.zeros((sequence_count, 200, 21), dtype=np.float32)
    feature_onehot = np.zeros((sequence_count, 200, 21), dtype=np.float32)
    feature_aac = np.zeros((sequence_count, 21), dtype=np.float32)

    index_out = np.zeros((sequence_count), dtype=np.int32)
    pssm_list, pssm_file_list = getpssmlist(pssmdir)
    #print(pssm_file_list)
    pssm_list_sort = []
    for i in range(sequence_count):
        str1 = sequence_data.iloc[i,0]
        for j in pssm_list:
            str2 = ''.join(j.iloc[:, 0].values)
            if str1 == str2:
                pssm_list_sort.append(j)
                break

    length_list = []
    for i in range(sequence_count):
        #length_list.append(len(sequence_data.iloc[i, 0])/sequence_data.iloc[i, 2])    #------------------------
        length_list.append(len(sequence_data.iloc[i, 0]))
    count_1 = 0
    for i in range(sequence_count):
        one_sequence = sequence_data.iloc[i,0]
        index_out[i], feature_pssm[i], sequence_pssm = psssmfeature(i, one_sequence, pssm_list_sort)
        feature_onehot[i] = one_hot_vector(i, one_sequence)
        feature_aac[i] = AAC_ft(i, one_sequence)
        count_1 = count_1 +1
    print('total    ',count_1)
    return index_out, length_list, feature_pssm, feature_onehot, feature_aac
