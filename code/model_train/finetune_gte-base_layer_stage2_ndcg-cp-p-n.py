import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from MAGDataset import MAGDataset_cpVSp
from functions import average_pool, loss_func_cpVSpVSn_ndcg

train_batch_size=96
val_batch_size=50
top_k=5
num_epochs=10
random_seed=2024
lr=1e-5
num_layer=5

train_link_table_name='train_table'
val_link_table_name='val_table'
test_link_table_name='test_table'

time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
device = 'cuda'

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

pretrained_model_dir=os.path.join('..','..','model_pretrain')
# previous_model_dir=os.path.join('..','..','model','s1_ckpt_gte-base_5layer_plus_2024-02-22_13-49-24')
# previous_model_index=8
# previous_model_dir=os.path.join('..','..','model','s1_ckpt_gte-base_5layer_plus_2024-02-14_11-23-21')
# previous_model_index=9
# previous_model_dir=os.path.join('..','..','model','s1_ckpt_gte-base_5layer_plus_2024-02-13_10-38-31')
# previous_model_index=5

model_dir=os.path.join('..','..','model')
ckpt_dir=os.path.join(model_dir,f's2_ndcg-cp-p-n_ckpt_gte-base_{num_layer}layer_plus_{time_data}')
os.makedirs(ckpt_dir,exist_ok=True)

config_dict={'train_batch_size':train_batch_size,'val_batch_size':val_batch_size,'top_k':top_k,'num_epochs':num_epochs,'lr':lr,'num_layer':num_layer,'train_link_table_name':train_link_table_name,'val_link_table_name':val_link_table_name,'test_link_table_name':test_link_table_name,'time_data':time_data}
# config_dict={'train_batch_size':train_batch_size,'val_batch_size':val_batch_size,'top_k':top_k,'num_epochs':num_epochs,'lr':lr,'num_layer':num_layer,'train_link_table_name':train_link_table_name,'val_link_table_name':val_link_table_name,'test_link_table_name':test_link_table_name,'previous_model_dir':previous_model_dir,'previous_model_index':previous_model_index,'time_data':time_data}
with open(os.path.join(ckpt_dir,'config.txt'),'w') as f:
    f.write(str(config_dict))

train_link_table=os.path.join('..','..','data','dataset_new',train_link_table_name+'.npy')
val_link_table=os.path.join('..','..','data','dataset_new',val_link_table_name+'.npy')
test_link_table=os.path.join('..','..','data','dataset_new',test_link_table_name+'.npy')

file_dir=os.path.join('..','..','data','dataset_new','text')
train_data=MAGDataset_cpVSp(file_dir,train_link_table)
train_loader=DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=True)
val_data=MAGDataset_cpVSp(file_dir,val_link_table)
val_loader=DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=False, num_workers=0, drop_last=True)

model = AutoModel.from_pretrained(os.path.join(pretrained_model_dir,"thenlper/gte-base"))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_model_dir,"thenlper/gte-base"))
model=torch.nn.DataParallel(model)
# model.load_state_dict(torch.load(os.path.join(previous_model_dir,f'model_{previous_model_index}.pth')))
model=model.to(device)

for name in model.module.embeddings.state_dict():
    layer=eval('model.module.embeddings.'+name)
    layer.requires_grad=False

for i in range(12-num_layer):
    for name in model.module.encoder.layer[i].state_dict():
        layer=eval(f'model.module.encoder.layer[(i)].'+name)
        layer.requires_grad=False

optimizer = torch.optim.Adam(model.module.encoder.layer[-num_layer:].parameters(), lr=lr)

train_prec_cp_record=list()
train_prec_p_record=list()
train_loss_record=list()

val_prec_cp_record=list()
val_prec_p_record=list()
for i in range(num_epochs):
    model.train()
    train_prec_cp=list()
    train_prec_p=list()
    train_loss=list()
    for batch in tqdm(train_loader):
        paper_q,paper_cp1,paper_cp2,paper_cp3,paper_cp4,paper_cp5,paper_p1,paper_p2,paper_p3,paper_p4,paper_p5=batch
        paper_k=paper_cp1+paper_cp2+paper_cp3+paper_cp4+paper_cp5+paper_p1+paper_p2+paper_p3+paper_p4+paper_p5

        token_q=tokenizer(paper_q, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        token_k=tokenizer(paper_k, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        
        outputs_q=model(**token_q)
        outputs_k=model(**token_k)
        
        embeddings_q = average_pool(outputs_q.last_hidden_state, token_q['attention_mask'])
        embeddings_k = average_pool(outputs_k.last_hidden_state, token_k['attention_mask'])
        
        prec_cp,prec_c,loss=loss_func_cpVSpVSn_ndcg(embeddings_q,embeddings_k,top_k)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_prec_cp+=prec_cp
        train_prec_p+=prec_c
        train_loss.append(loss.cpu().item())

    train_prec_cp_epoch=np.mean(train_prec_cp)
    train_prec_p_epoch=np.mean(train_prec_p)
    train_loss_epoch=np.mean(train_loss)
    train_prec_cp_record.append(train_prec_cp_epoch)
    train_prec_p_record.append(train_prec_p_epoch)
    train_loss_record.append(train_loss_epoch)
    print('Epoch %d, Train-Precision-CP=%.3f, Train-Precision-P=%.3f, Train-Loss=%.3f'%(i,train_prec_cp_epoch,train_prec_p_epoch,train_loss_epoch))

    model.eval()
    val_prec_cp=list()
    val_prec_p=list()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            paper_q,paper_cp1,paper_cp2,paper_cp3,paper_cp4,paper_cp5,paper_p1,paper_p2,paper_p3,paper_p4,paper_p5=batch
            paper_k=paper_cp1+paper_cp2+paper_cp3+paper_cp4+paper_cp5+paper_p1+paper_p2+paper_p3+paper_p4+paper_p5

            token_q=tokenizer(paper_q, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            token_k=tokenizer(paper_k, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            
            outputs_q=model(**token_q)
            outputs_k=model(**token_k)
            
            embeddings_q = average_pool(outputs_q.last_hidden_state, token_q['attention_mask'])
            embeddings_k = average_pool(outputs_k.last_hidden_state, token_k['attention_mask'])
            
            prec_cp,prec_c,loss=loss_func_cpVSpVSn_ndcg(embeddings_q,embeddings_k,top_k)
            val_prec_cp+=prec_cp
            val_prec_p+=prec_c

    val_prec_cp_epoch=np.mean(val_prec_cp)
    val_prec_p_epoch=np.mean(val_prec_p)
    val_prec_cp_record.append(val_prec_cp_epoch)
    val_prec_p_record.append(val_prec_p_epoch)
    print('Epoch %d, Val-Precision-CP=%.3f, Val-Precision-P=%.3f'%(i,val_prec_cp_epoch,val_prec_p_epoch))
    torch.save(model.state_dict(),os.path.join(ckpt_dir,f'model_{i}.pth'))

    np.save(os.path.join(ckpt_dir,f'train_prec_cp_record_{i}.npy'),np.array(train_prec_cp))
    np.save(os.path.join(ckpt_dir,f'train_prec_p_record_{i}.npy'),np.array(train_prec_p))
    np.save(os.path.join(ckpt_dir,f'train_loss_record_{i}.npy'),np.array(train_loss))

    np.save(os.path.join(ckpt_dir,'train_prec_cp_record.npy'),np.array(train_prec_cp_record))
    np.save(os.path.join(ckpt_dir,'train_prec_p_record.npy'),np.array(train_prec_p_record))
    np.save(os.path.join(ckpt_dir,'train_loss_record.npy'),np.array(train_loss_record))

    np.save(os.path.join(ckpt_dir,'val_prec_cp_record.npy'),np.array(val_prec_cp_record))
    np.save(os.path.join(ckpt_dir,'val_prec_p_record.npy'),np.array(val_prec_p_record))