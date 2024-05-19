from torch.utils import data
import os
import numpy as np

class MAGDataset(data.Dataset):
    def __init__(self,file_dir,link_table):
        self.file_dir=file_dir
        self.link_table=np.load(link_table)

    def __getitem__(self, index):
        paper_id1,paper_id2=self.link_table[index]
        paper_id1=str(paper_id1)
        paper_id2=str(paper_id2)

        paper1=open(os.path.join(self.file_dir+paper_id1[-1],f'{paper_id1}.txt'),'r').read()
        paper2=open(os.path.join(self.file_dir+paper_id2[-1],f'{paper_id2}.txt'),'r').read()
        
        return paper1,paper2

    def __len__(self):
        return len(self.link_table)
    
class MAGDataset_cpVSp(data.Dataset):
    def __init__(self,file_dir,link_table):
        self.file_dir=file_dir
        self.link_table=np.load(link_table)

    def __getitem__(self, index):
        paper_id,paper_cp_id1,paper_cp_id2,paper_cp_id3,paper_cp_id4,paper_cp_id5,paper_p_id1,paper_p_id2,paper_p_id3,paper_p_id4,paper_p_id5=self.link_table[index]

        paper_id=str(paper_id)
        paper_cp_id1=str(paper_cp_id1)
        paper_cp_id2=str(paper_cp_id2)
        paper_cp_id3=str(paper_cp_id3)
        paper_cp_id4=str(paper_cp_id4)
        paper_cp_id5=str(paper_cp_id5)
        paper_p_id1=str(paper_p_id1)
        paper_p_id2=str(paper_p_id2)
        paper_p_id3=str(paper_p_id3)
        paper_p_id4=str(paper_p_id4)
        paper_p_id5=str(paper_p_id5)

        paper=open(os.path.join(self.file_dir+paper_id[-1],f'{paper_id}.txt'),'r').read()
        paper_cp1=open(os.path.join(self.file_dir+paper_cp_id1[-1],f'{paper_cp_id1}.txt'),'r').read()
        paper_cp2=open(os.path.join(self.file_dir+paper_cp_id2[-1],f'{paper_cp_id2}.txt'),'r').read()
        paper_cp3=open(os.path.join(self.file_dir+paper_cp_id3[-1],f'{paper_cp_id3}.txt'),'r').read()
        paper_cp4=open(os.path.join(self.file_dir+paper_cp_id4[-1],f'{paper_cp_id4}.txt'),'r').read()
        paper_cp5=open(os.path.join(self.file_dir+paper_cp_id5[-1],f'{paper_cp_id5}.txt'),'r').read()
        paper_p1=open(os.path.join(self.file_dir+paper_p_id1[-1],f'{paper_p_id1}.txt'),'r').read()
        paper_p2=open(os.path.join(self.file_dir+paper_p_id2[-1],f'{paper_p_id2}.txt'),'r').read()
        paper_p3=open(os.path.join(self.file_dir+paper_p_id3[-1],f'{paper_p_id3}.txt'),'r').read()
        paper_p4=open(os.path.join(self.file_dir+paper_p_id4[-1],f'{paper_p_id4}.txt'),'r').read()
        paper_p5=open(os.path.join(self.file_dir+paper_p_id5[-1],f'{paper_p_id5}.txt'),'r').read()
        
        return paper,paper_cp1,paper_cp2,paper_cp3,paper_cp4,paper_cp5,paper_p1,paper_p2,paper_p3,paper_p4,paper_p5

    def __len__(self):
        return len(self.link_table)