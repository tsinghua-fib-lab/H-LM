import os
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm

from functions import test_cp_logits_topk
from sklearn.metrics import ndcg_score

ckpt_model_dir=os.path.join('..','model','s3_analyse-rank')
indice_name_list=[
    'test_indices_GPT3.5_SINGLEexample_analyse+rank3in6_batch0-49-candidate10000-top10.npy',

    'test_indices_GPT3.5_SINGLEexample_rank3in6_batch0-49-candidate10000-top10.npy',
    'test_indices_GPT3.5_NONEexample_analyse+rank3in6_batch0-49-candidate10000-top10.npy',
    'test_indices_GPT3.5_NONEexample_rank3in6_batch0-49-candidate10000-top10.npy',

    'test_indices_GPT3.5_MULTIexample_analyse+rank3in6_batch0-4-candidate10000-top10.npy',

    'test_indices_GPT3.5_SINGLEexample_analyse+rank1in2_batch0-4-candidate10000-top10.npy',
    'test_indices_GPT3.5_SINGLEexample_analyse+rank2in4_batch0-4-candidate10000-top10.npy',
    'test_indices_GPT3.5_SINGLEexample_analyse+rank3in6_batch0-4-candidate10000-top10.npy',
    'test_indices_GPT3.5_SINGLEexample_analyse+rank4in8_batch0-4-candidate10000-top10.npy',
    'test_indices_GPT3.5_SINGLEexample_analyse+rank5in10_batch0-4-candidate10000-top10.npy',

    'test_indices_GPT4o_SINGLEexample_analyse+rank3in6_batch0-4-candidate10000-top10.npy',
    'test_indices_GPT4_SINGLEexample_analyse+rank3in6_batch0-4-candidate10000-top10.npy',
    'test_indices_Llama3-70B_SINGLEexample_analyse+rank3in6_batch0-4-candidate10000-top10.npy',
    'test_indices_Llama3-8B_SINGLEexample_analyse+rank3in6_batch0-4-candidate10000-top10.npy',
    'test_indices_Mixtral-8×22B_SINGLEexample_analyse+rank3in6_batch0-4-candidate10000-top10.npy',
    'test_indices_Mixtral-8×7B_SINGLEexample_analyse+rank3in6_batch0-4-candidate10000-top10.npy',
    'test_indices_ChatGLM2-6B_SINGLEexample_analyse+rank3in6_batch0-4-candidate10000-top10.npy',
    ]

batch_size=1000
candidate_size=batch_size*10

test_table=np.load(os.path.join('..','data','dataset_new','test_table.npy'))
cp_score=pickle.load(open(os.path.join('..','data','dataset_new','paper_core_citation_score'),'rb'))

for indice_name in indice_name_list:
    indices=np.load(os.path.join(ckpt_model_dir,indice_name))

    num_paper=indices.shape[0]
    scores=torch.zeros(num_paper,num_paper*10)
    scores_cp=torch.zeros(num_paper,num_paper*10)
    scores_p=torch.zeros(num_paper,num_paper*10)
    for i in range(num_paper):
        for j in range(5):
            scores[i,i+j*num_paper]=cp_score[test_table[i][0]][test_table[i][j+1]]
            scores_cp[i,i+j*num_paper]=1
        for j in range(10):
            scores_p[i,i+j*num_paper]=1

    result_dict=dict()

    test_ndcg_cp_3=list()
    test_ndcg_cp_5=list()

    test_prec_cp_3=list()
    test_prec_cp_5=list()

    for batch_count in tqdm(range(num_paper//batch_size)):
        col_index=[batch_count*batch_size + x + y * num_paper for y in range(10) for x in range(batch_size)]
        batch_logits=torch.zeros(batch_size,batch_size*10)
        for i in range(batch_size):
            for j in range(10):
                batch_logits[i,indices[batch_count*batch_size+i,j]]=10-j
        batch_scores=scores[batch_count*batch_size:(batch_count+1)*batch_size,col_index]
        batch_scores_cp=scores_cp[batch_count*batch_size:(batch_count+1)*batch_size,col_index]
        batch_scores_p=scores_p[batch_count*batch_size:(batch_count+1)*batch_size,col_index]

        for i in range(batch_size):
            test_ndcg_cp_3.append(ndcg_score([batch_scores_cp[i]],[batch_logits[i]],k=3))
            test_ndcg_cp_5.append(ndcg_score([batch_scores_cp[i]],[batch_logits[i]],k=5))

        test_prec_cp_3.extend(test_cp_logits_topk(batch_logits,3))
        test_prec_cp_5.extend(test_cp_logits_topk(batch_logits,5))

    ndcg_cp_3=np.mean(test_ndcg_cp_3)
    ndcg_cp_5=np.mean(test_ndcg_cp_5)

    prec_cp_3=np.mean(test_prec_cp_3)
    prec_cp_5=np.mean(test_prec_cp_5)


    print(indice_name)
    print(f'candidate10000 | ndcg_cp_3: {ndcg_cp_3}, ndcg_cp_5: {ndcg_cp_5}, prec_cp_3: {prec_cp_3}, prec_cp_5: {prec_cp_5}')

    result_dict['ndcg_cp_3']=ndcg_cp_3
    result_dict['ndcg_cp_5']=ndcg_cp_5
    result_dict['prec_cp_3']=prec_cp_3
    result_dict['prec_cp_5']=prec_cp_5

    with open(os.path.join(ckpt_model_dir,indice_name.replace('.npy','_final.json')),'w') as f:
        json.dump(result_dict,f)

    with open(os.path.join(ckpt_model_dir,indice_name.replace('.npy','_prec_cp_3.pkl')),'wb') as f:
        pickle.dump(test_prec_cp_3,f)
    with open(os.path.join(ckpt_model_dir,indice_name.replace('.npy','_prec_cp_5.pkl')),'wb') as f:
        pickle.dump(test_prec_cp_5,f)
    
    with open(os.path.join(ckpt_model_dir,indice_name.replace('.npy','_ndcg_cp_3.pkl')),'wb') as f:
        pickle.dump(test_ndcg_cp_3,f)
    with open(os.path.join(ckpt_model_dir,indice_name.replace('.npy','_ndcg_cp_5.pkl')),'wb') as f:
        pickle.dump(test_ndcg_cp_5,f)
    