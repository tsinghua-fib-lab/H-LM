import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from MAGDataset import MAGDataset
from functions import average_pool, loss_func

train_batch_size=512
val_batch_size=512
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
model_dir=os.path.join('..','..','model')
ckpt_dir=os.path.join(model_dir,f's1_ckpt_gte-base_{num_layer}layer_plus_{time_data}')
os.makedirs(ckpt_dir,exist_ok=True)

config_dict={'train_batch_size':train_batch_size,'val_batch_size':val_batch_size,'num_epochs':num_epochs,'lr':lr,'num_layer':num_layer,'train_link_table_name':train_link_table_name,'val_link_table_name':val_link_table_name,'test_link_table_name':test_link_table_name,'time_data':time_data}
with open(os.path.join(ckpt_dir,'config.txt'),'w') as f:
    f.write(str(config_dict))

train_link_table=os.path.join('..','..','data','dataset',train_link_table_name+'.npy')
val_link_table=os.path.join('..','..','data','dataset',val_link_table_name+'.npy')

file_dir=os.path.join('..','..','data','dataset','text')
train_data=MAGDataset(file_dir,train_link_table)
train_loader=DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=True)
val_data=MAGDataset(file_dir,val_link_table)
val_loader=DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=False, num_workers=0, drop_last=True)

model = AutoModel.from_pretrained(os.path.join(pretrained_model_dir,"thenlper/gte-base"))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_model_dir,"thenlper/gte-base"))
model=torch.nn.DataParallel(model)
model=model.to(device)

for name in model.module.embeddings.state_dict():
    layer=eval('model.module.embeddings.'+name)
    layer.requires_grad=False

for i in range(12-num_layer):
    for name in model.module.encoder.layer[i].state_dict():
        layer=eval(f'model.module.encoder.layer[(i)].'+name)
        layer.requires_grad=False

optimizer = torch.optim.Adam(model.module.encoder.layer[-num_layer:].parameters(), lr=lr)

train_prec_record=list()
train_loss_record=list()

val_prec_record=list()
for i in range(num_epochs):
    model.train()
    train_prec=list()
    train_loss=list()
    for batch in tqdm(train_loader):
        paper_q,paper_k=batch

        token_q=tokenizer(paper_q, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        token_k=tokenizer(paper_k, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

        outputs_q=model(**token_q)
        outputs_k=model(**token_k)
        
        embeddings_q = average_pool(outputs_q.last_hidden_state, token_q['attention_mask'])
        embeddings_k = average_pool(outputs_k.last_hidden_state, token_k['attention_mask'])
        
        prec,loss=loss_func(embeddings_q,embeddings_k)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_prec+=prec
        train_loss.append(loss.cpu().item())

    train_prec_epoch=np.mean(train_prec)
    train_loss_epoch=np.mean(train_loss)
    train_prec_record.append(train_prec_epoch)
    train_loss_record.append(train_loss_epoch)
    print('Epoch %d, Train-Precision=%.3f, Train-Loss=%.3f'%(i,train_prec_epoch,train_loss_epoch))

    model.eval()
    val_prec=list()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            paper_q,paper_k=batch
            token_q=tokenizer(paper_q, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            token_k=tokenizer(paper_k, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            
            outputs_q=model(**token_q)
            outputs_k=model(**token_k)
            
            embeddings_q = average_pool(outputs_q.last_hidden_state, token_q['attention_mask'])
            embeddings_k = average_pool(outputs_k.last_hidden_state, token_k['attention_mask'])
            
            prec,loss=loss_func(embeddings_q,embeddings_k)
            val_prec+=prec

    val_prec_epoch=np.mean(val_prec)
    val_prec_record.append(val_prec_epoch)
    print('Epoch %d, Val-Precision=%.3f'%(i,val_prec_epoch))
    torch.save(model.state_dict(),os.path.join(ckpt_dir,f'model_{i}.pth'))

    np.save(os.path.join(ckpt_dir,f'train_prec_record_{i}.npy'),np.array(train_prec))
    np.save(os.path.join(ckpt_dir,f'train_loss_record_{i}.npy'),np.array(train_loss))

    np.save(os.path.join(ckpt_dir,'train_prec_record.npy'),np.array(train_prec_record))
    np.save(os.path.join(ckpt_dir,'train_loss_record.npy'),np.array(train_loss_record))

    np.save(os.path.join(ckpt_dir,'val_prec_record.npy'),np.array(val_prec_record))