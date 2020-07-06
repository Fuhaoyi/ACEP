import os

def command_pssm(blast_path, database_nr, one_query, out_file, out_pssm):
    os.system('%s \
                -db %s \
                -query %s \
                -num_iterations 3 \
                -evalue 1e-3 \
                -out %s \
                -out_ascii_pssm %s' %(blast_path,database_nr, one_query, out_file, out_pssm))

def getOneSequence(query_seqs):
    input_file = open(query_seqs,'r')
    line_list = input_file.readlines()
    input_file.close()
    i = 0
    count = 1
    while i < len(line_list):
        filename = "{:0>5d}".format(count)+'.fasta'
        output_file = open('sequences/%s'%filename, 'w')
        str_write = line_list[i]+line_list[i+1]
        output_file.write(str_write)
        output_file.close()
        i = i+2
        count = count+1

def callBLASTgetPSSMs(path0, path1, path2):
    getOneSequence('queryseq.fasta')
    dir = 'sequences/'
    dir_list = os.listdir(dir)
    for i in dir_list:
        print(i)
        one_query_name = path2 + 'sequences/' + i
        out_file_name =path2 + 'results/' + i.split('.')[0]+'.out'
        out_pssm_name = path2 + 'pssms/' + i.split('.')[0] + '.pssm'
        command_pssm(path0, path1, one_query_name, out_file_name, out_pssm_name)
        print(out_pssm_name)

callBLASTgetPSSMs(path0 ='E:/blast-2.9.0+/bin/psiblast',
                  path1 ='E:/blast-2.9.0+/bin/uniref90.db',
                  path2 ='E:/ACEP/ACME_codes/localBLASTgetPSSMs/')