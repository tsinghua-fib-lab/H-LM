import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm

from functions import test_prec_record_logits


ckpt_model_name=sys.argv[1]
ckpt_model_index=int(sys.argv[2])
ckpt_model_dir=os.path.join('..','..','model',ckpt_model_name)
# stage1 s1_ckpt_gte-base_5layer_plus_2024-02-14_11-23-21 9
# stage2 s2_ndcg-cp-p-n_ckpt_gte-base_5layer_plus_2024-02-22_21-29-31 3
# stage1+stage2 s2_ndcg-cp-p-n_ckpt_gte-base_5layer_plus_2024-02-21_21-22-33 3

print(f'{ckpt_model_name}-{ckpt_model_index}')

embeddings=np.load(os.path.join(ckpt_model_dir,f'test_embeddings_{ckpt_model_index}.npy'))
num_paper=embeddings.shape[0]

embeddings_q = torch.tensor(embeddings[:,0,:])
embeddings_k = torch.tensor(embeddings[:,1:,:]).permute(1,0,2).reshape(num_paper*10,-1)
logits=torch.matmul(embeddings_q, embeddings_k.transpose(0, 1))

batch_size=1000
candidate_size=batch_size*10

test_prec_cp_5=list()
test_prec_p_5=list()
test_prec_p_10=list()

indices_record=list()
for batch_count in range(num_paper//batch_size):
    col_index=[batch_count*batch_size + x + y * num_paper for y in range(10) for x in range(batch_size)]
    batch_logits=logits[batch_count*batch_size:(batch_count+1)*batch_size,col_index]
    
    prec_cp_5,prec_p_5,prec_p_10,indices=test_prec_record_logits(batch_logits)
    test_prec_cp_5.extend(prec_cp_5)
    test_prec_p_5.extend(prec_p_5)
    test_prec_p_10.extend(prec_p_10)

    indices_record.extend(indices)

np.save(os.path.join(ckpt_model_dir,f'test_indices_{ckpt_model_index}-candidate{candidate_size}-top10.npy'),np.array(indices_record))

prec_cp_5=np.mean(test_prec_cp_5)
prec_p_5=np.mean(test_prec_p_5)
prec_p_10=np.mean(test_prec_p_10)
print(f'candidate{candidate_size} | prec_cp_5: {prec_cp_5}, prec_p_5: {prec_p_5}, prec_p_10: {prec_p_10}')