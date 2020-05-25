import pandas as pd
import numpy as np
import os


def getallseq():
    train = pd.read_csv('amp.tr.all.csv')
    eval = pd.read_csv('amp.eval.all.csv')
    test = pd.read_csv('amp.te.all.csv')

    seq_all = pd.concat([train, eval, test], axis = 0, ignore_index=True)
    seq_all.to_csv('concat_allseq.csv')

    seq_repeat = seq_all
    col = seq_repeat.columns.insert(2, 'repeat')
    seq_repeat = seq_repeat.reindex(columns=col, fill_value=0)

    for i in range(seq_repeat.shape[0]):
        one_sequence = seq_repeat.iloc[i,0]
        k = 1
        while(len(one_sequence)*k<60):
            k = k+1
        one_sequence_k = one_sequence*k
        seq_repeat.iloc[i, 0] = one_sequence_k
        seq_repeat.iloc[i, 2] = k

    seq_repeat.to_csv('seq_repeat_k.csv')

    seq_list = []
    for i in range(seq_repeat.shape[0]):
        seq_list.append(seq_repeat.iloc[i,0])

    fo = open("seq_repeat.fasta", "w")
    count=0
    for i in seq_list:
        fo.write('>seq'+str(count)+'_'+str(seq_repeat.iloc[count,2])+'\n')
        fo.write(i+'\n')
        count = count+1
    fo.close()

#getallseq()


def cutpssm(filename,out_name,k):
    f = open(filename, 'r')
    linelist = f.readlines()
    #print(linelist)
    linelist_cut = []
    cut_point = int((len(linelist)-9)/k+3)
    for i in range(cut_point):
        linelist_cut.append(linelist[i])
    f.close()

    f2 = open(out_name, 'w')
    f2.writelines(linelist_cut)
    f2.close()

#cutpssm('6e60daedd6bc8a5654f41c016ec0593f_1.pssm','pssm_30_200\seq_30_60_1.pssm')


def getpssmlist(dir):
    filenamelist = os.listdir(dir)
    seq_repeat_k = pd.read_csv('seq_repeat_k.csv', index_col=0)
    #print(seq_repeat_k)
    #print(filenamelist)
    for i in filenamelist:
        filename = os.path.join(dir,i)
        pssm_id = i.split('_', 1)[1]
        pssm_id = pssm_id.split('.', 1)[0]
        pssm_id = int(pssm_id)
        pssm_id = pssm_id-1+3500
        #k = seq_repeat_k.iloc[pssm_id,2]
        k = 1
        out_name = '{:0>4d}'.format(pssm_id)+'.pssm'
        out_path =os.path.join('all_pssm_data','pssm_files0_3555',out_name)
        cutpssm(filename,out_path,k)

getpssmlist('all_pssm_data\pssm_files3500_3555')


def renamefun(dir):
    filenamelist = os.listdir(dir)
    for i in filenamelist:
        filename = os.path.join(dir,i)
        f1 = open(filename, 'r')
        linelist = f1.readlines()
        f1.close()

        rename1 = i.split('.', 1)[0]
        rename1 = int(rename1)
        rename1 = '{:0>4d}'.format(rename1)+'.pssm'

        out_path =os.path.join('all_pssm_data','pssm_files0_3555_cut','train_rename',rename1)
        f2 = open(out_path, 'w')
        f2.writelines(linelist)
        f2.close()

#renamefun(os.path.join('all_pssm_data','pssm_files0_3555_cut','train'))


def proccess2csv():
    f = open('DECOY.eval.fa', 'r')
    str1 = f.read()
    f.close()
    list1 = str1.split()
    list2 = []
    for i in list1:
        if i[0] != '>':
            list2.append(i)
    print(list2)
    print(len(list2))

    ng_ps = 0
    dt = pd.DataFrame({'seq': list2, 'label': [ng_ps] * len(list2)})
    dt = dt.reindex(columns=['seq', 'label'])
    print(dt)

    dt.to_csv('amp.eval.ng.res.csv', index=False, header=True)


def get_seq_fasta():
    seq_all = pd.read_csv('pssm_files0_3555_all\seq_all_data.csv',index_col=0)
    print(seq_all)
    seq_list = []
    for i in range(seq_all.shape[0]):
        seq_list.append(seq_all.iloc[i,0])

    fo = open("pssm_files0_3555_all\seq_all_fasta.fasta", "w")
    count=0
    for i in seq_list:
        fo.write('>seq'+str(count)+'\n')
        fo.write(i+'\n')
        count = count+1
    fo.close()
