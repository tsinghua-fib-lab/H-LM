# import setproctitle 
# setproctitle.setproctitle('Citation@haoqianyue')
#batch offset 1
import os
import pickle
import re
import sys
import time
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from openai import OpenAI

from MAGDataset import MAGDataset_cpVSp

class npencoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,np.integer):
            return int(obj)

system_prompt="""
Now you are a sophisticated researcher and information analyst, and going to investigate the problem of a specific paper citation. Your analysis should be based on the following steps:
Explore citation conventions and standards in academic fields. For example, citation serve to acknowledge prior work, provide evidence or support, facilitate further exploration and allow readers to trace the development and history of ideas or methodologies.
"""

analyzer_example="""
<question>
Here is the title and abstract of the query paper. 

Title: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
Abstract: Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the setting of language modeling. We propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism and a novel positional encoding scheme. Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem. As a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+ times faster than vanilla Transformers during evaluation. Notably, we improve the state-of-the-art results of bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on Penn Treebank (without finetuning). When trained only on WikiText-103, Transformer-XL manages to generate reasonably coherent, novel text articles with thousands of tokens. Our code, pretrained models, and hyperparameters are available in both Tensorflow and PyTorch.

Now you are doing a research following up this paper above. Here are some other research papers which have been already cited by the query paper.

0
Title: Attention is All you Need
Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU.

1
Title: Self-attention with relative position representations
Abstract: Relying entirely on an attention mechanism, the Transformer introduced by Vaswani et al. (2017) achieves state-of-the-art results for machine translation. In contrast to recurrent and convolutional neural networks, it does not explicitly model relative or absolute position information in its structure. Instead, it requires adding representations of absolute positions to its inputs. In this work we present an alternative approach, extending the self-attention mechanism to efficiently consider representations of the relative positions, or distances between sequence elements.

2
Title: Character-Level Language Modeling with Deeper Self-Attention
Abstract: LSTMs and other RNN variants have shown strong performance on character-level language modeling. These models are typically trained using truncated backpropagation through time, and it is common to assume that their success stems from their ability to remember long-term contexts. In this paper, we show that a deep (64-layer) transformer model (Vaswani et al. 2017) with fixed context outperforms RNN variants by a large margin, achieving state of the art on two popular benchmarks: 1.13 bits per character on text8 and 1.06 on enwik8. To get good results at this depth, we show that it is important to add auxiliary losses, both at intermediate network layers and intermediate sequence positions.

3
Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Abstract: We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

4
Title: Adaptive input representations for neural language modelingC
Abstract: We introduce adaptive input representations for neural language modeling which extend the adaptive softmax of Grave et al. (2017) to input representations of variable capacity. There are several choices on how to factorize the input and output layers, and whether to model words, characters or sub-word units. We perform a systematic comparison of popular choices for a self-attentional architecture. Our experiments show that models equipped with adaptive embeddings are more than twice as fast to train than the popular character input CNN while having a lower number of parameters. On the WikiText-103 benchmark we achieve 18.7 perplexity, an improvement of 10.5 perplexity compared to the previously best published result and on the Billion Word benchmark, we achieve 23.02 perplexity.

5
Title: A Neural Probabilistic Language Model
Abstract: A goal of statistical language modeling is to learn the joint probability function of sequences of words. This is intrinsically difficult because of the curse of dimensionality: we propose to fight it with its own weapons. In the proposed approach one learns simultaneously (1) a distributed rep(cid:173) resentation for each word (i.e. a similarity between words) along with (2) the probability function for word sequences, expressed with these repre(cid:173) sentations. Generalization is obtained because a sequence of words that has never been seen before gets high probability if it is made of words that are similar to words forming an already seen sentence.

Try to think abductively and convince yourself as a researcher. Figure out why the query paper cite these one by one. Try to think step by step before giving the answer. 

</question>

<answer>
To understand why the query paper, "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context," cites each of the listed papers, we need to consider the contributions and relevance of each cited work to the development, context, or findings of the query paper. Let's analyze them one by one:

1. **Attention is All You Need** (Paper 0)
   - **Relevance**: This paper introduces the Transformer model, which is the foundation upon which Transformer-XL builds. The Transformer model revolutionized natural language processing (NLP) by moving away from recurrent and convolutional networks, focusing instead on attention mechanisms to process sequences of data. The query paper extends the Transformer model to handle longer contexts, which is a direct expansion of the work introduced in this paper.
   - **Reason for Citation**: To acknowledge the foundational model (Transformer) on which Transformer-XL is based and to discuss the limitations of the original Transformer model that the query paper aims to overcome.

2. **Self-attention with relative position representations** (Paper 1)
   - **Relevance**: The introduction of relative position representations in self-attention mechanisms is a key innovation that allows Transformers to better understand the relationships between different parts of a sequence. This concept is important for the Transformer-XL, which seeks to improve the model's ability to handle long-term dependencies.
   - **Reason for Citation**: To discuss advancements in self-attention mechanisms that are relevant to the development of Transformer-XL, particularly the handling of position information in sequences, which is crucial for modeling longer contexts.

3. **Character-Level Language Modeling with Deeper Self-Attention** (Paper 2)
   - **Relevance**: This paper demonstrates the effectiveness of deep Transformer models for character-level language modeling, highlighting the potential of Transformers to capture complex dependencies in text. It sets a precedent for using Transformers in language modeling, particularly at the character level, and discusses techniques to improve their performance.
   - **Reason for Citation**: To support the argument that deep Transformer models, like Transformer-XL, are effective at language modeling tasks and to build on the idea of enhancing Transformer architectures for better performance in NLP tasks.

4. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (Paper 3)
   - **Relevance**: BERT represents a significant leap forward in pre-training language representations, using a bidirectional Transformer. While BERT focuses on understanding language context in both directions, Transformer-XL aims to extend the context length that models can effectively process.
   - **Reason for Citation**: To highlight the importance of deep bidirectional Transformers in NLP and to position Transformer-XL within the broader context of recent advancements in Transformer-based models. It may also cite BERT to discuss differences in approach, particularly regarding context length and model architecture.

5. **Adaptive input representations for neural language modeling** (Paper 4)
   - **Relevance**: This paper explores adaptive input representations, which can make models more efficient and effective by adjusting the capacity of input representations based on the complexity of the input. Such techniques are relevant for Transformer-XL, which seeks to improve efficiency and performance in language modeling.
   - **Reason for Citation**: To discuss methods for improving the efficiency of neural language models, particularly in the context of Transformer-based architectures. The query paper might leverage or build upon these adaptive techniques to enhance Transformer-XL's performance.

6. **A Neural Probabilistic Language Model** (Paper 5)
   - **Relevance**: This work is foundational in the field of neural language modeling, introducing the concept of learning distributed representations for words alongside the probability function for word sequences. It lays the groundwork for subsequent developments in language modeling, including the use of Transformers.
   - **Reason for Citation**: To acknowledge the historical context and evolution of language modeling techniques leading up to the development of Transformer and Transformer-XL models. It may also cite this work to discuss the importance of distributed representations in understanding language.
</answer>


"""
ranker_example="""
<question>
Here is the title and abstract of the query paper.

Title: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
Abstract: Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the setting of language modeling. We propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism and a novel positional encoding scheme. Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem. As a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+ times faster than vanilla Transformers during evaluation. Notably, we improve the state-of-the-art results of bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on Penn Treebank (without finetuning). When trained only on WikiText-103, Transformer-XL manages to generate reasonably coherent, novel text articles with thousands of tokens. Our code, pretrained models, and hyperparameters are available in both Tensorflow and PyTorch.

There are some other candidate papers and the analysis of why this query paper cites these.

<analysis>
To understand why the query paper, "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context," cites each of the listed papers, we need to consider the contributions and relevance of each cited work to the development, context, or findings of the query paper. Let's analyze them one by one:

1. **Attention is All You Need** (Paper 0)
   - **Relevance**: This paper introduces the Transformer model, which is the foundation upon which Transformer-XL builds. The Transformer model revolutionized natural language processing (NLP) by moving away from recurrent and convolutional networks, focusing instead on attention mechanisms to process sequences of data. The query paper extends the Transformer model to handle longer contexts, which is a direct expansion of the work introduced in this paper.
   - **Reason for Citation**: To acknowledge the foundational model (Transformer) on which Transformer-XL is based and to discuss the limitations of the original Transformer model that the query paper aims to overcome.

2. **Self-attention with relative position representations** (Paper 1)
   - **Relevance**: The introduction of relative position representations in self-attention mechanisms is a key innovation that allows Transformers to better understand the relationships between different parts of a sequence. This concept is important for the Transformer-XL, which seeks to improve the model's ability to handle long-term dependencies.
   - **Reason for Citation**: To discuss advancements in self-attention mechanisms that are relevant to the development of Transformer-XL, particularly the handling of position information in sequences, which is crucial for modeling longer contexts.

3. **Character-Level Language Modeling with Deeper Self-Attention** (Paper 2)
   - **Relevance**: This paper demonstrates the effectiveness of deep Transformer models for character-level language modeling, highlighting the potential of Transformers to capture complex dependencies in text. It sets a precedent for using Transformers in language modeling, particularly at the character level, and discusses techniques to improve their performance.
   - **Reason for Citation**: To support the argument that deep Transformer models, like Transformer-XL, are effective at language modeling tasks and to build on the idea of enhancing Transformer architectures for better performance in NLP tasks.

4. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (Paper 3)
   - **Relevance**: BERT represents a significant leap forward in pre-training language representations, using a bidirectional Transformer. While BERT focuses on understanding language context in both directions, Transformer-XL aims to extend the context length that models can effectively process.
   - **Reason for Citation**: To highlight the importance of deep bidirectional Transformers in NLP and to position Transformer-XL within the broader context of recent advancements in Transformer-based models. It may also cite BERT to discuss differences in approach, particularly regarding context length and model architecture.

5. **Adaptive input representations for neural language modeling** (Paper 4)
   - **Relevance**: This paper explores adaptive input representations, which can make models more efficient and effective by adjusting the capacity of input representations based on the complexity of the input. Such techniques are relevant for Transformer-XL, which seeks to improve efficiency and performance in language modeling.
   - **Reason for Citation**: To discuss methods for improving the efficiency of neural language models, particularly in the context of Transformer-based architectures. The query paper might leverage or build upon these adaptive techniques to enhance Transformer-XL's performance.

6. **A Neural Probabilistic Language Model** (Paper 5)
   - **Relevance**: This work is foundational in the field of neural language modeling, introducing the concept of learning distributed representations for words alongside the probability function for word sequences. It lays the groundwork for subsequent developments in language modeling, including the use of Transformers.
   - **Reason for Citation**: To acknowledge the historical context and evolution of language modeling techniques leading up to the development of Transformer and Transformer-XL models. It may also cite this work to discuss the importance of distributed representations in understanding language.
</analysis>

Now you are doing a research following up this query paper (Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context) above. 
Use the analysis to identify patterns or themes that suggest potential citation relationships.
Rank these candidate papers in the order you are most likely to cite from the perspective of a research follower and provide explanations or justifications for your reasoning.
After finishing your analysis with ranking result, please answer an extra single line in the end following the format {ranked order: paper 0/1/2/3/4/5, paper 0/1/2/3/4/5, paper 0/1/2/3/4/5, paper 0/1/2/3/4/5, paper 0/1/2/3/4/5, paper 0/1/2/3/4/5} (one number each) and do not include any other words.

</question>

<answer>
Given the context and the analysis provided, the ranking of the candidate papers from most likely to least likely to be cited in a follow-up research to "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" would be based on their direct relevance to the advancements made by Transformer-XL, their foundational role in the development of Transformer models, and their contribution to the broader field of language modeling and neural network efficiency. Here's the ranking with justifications:

1. **Attention is All You Need (Paper 0)**: This paper is the cornerstone of Transformer models. Any research following Transformer-XL would likely reference this to acknowledge the foundational model and its limitations that the follow-up work seeks to address or build upon.

2. **Self-attention with relative position representations (Paper 1)**: The methodological relevance of improving self-attention mechanisms, especially for handling longer contexts in Transformer models, makes this paper a critical citation for discussing technical advancements or modifications in a follow-up study.

3. **Character-Level Language Modeling with Deeper Self-Attention (Paper 2)**: This paper's focus on deep Transformer models for character-level language modeling aligns closely with the objectives of Transformer-XL, making it a likely citation for discussions on model depth and granularity in language modeling.

4. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Paper 3)**: Given the significant impact of BERT on the NLP field and its methodological similarities and differences with Transformer-XL, a follow-up study would likely cite it to discuss further advancements or comparisons in Transformer-based model architectures.

5. **Adaptive input representations for neural language modeling (Paper 4)**: Techniques for improving model efficiency and input representation are crucial for advancing Transformer models. A follow-up study might cite this work to explore or introduce new adaptive techniques for enhancing Transformer-XL's efficiency or performance.

6. **A Neural Probabilistic Language Model (Paper 5)**: While foundational to the field of neural language modeling, this paper might be cited less frequently in a direct follow-up to Transformer-XL, except to provide historical context or discuss the evolution of language modeling techniques leading up to Transformer models.

{ranked order: paper 0, paper 1, paper 2, paper 3, paper 4, paper 5}
</answer>
"""


batch_size=1000
candidate_size=10*batch_size
test_link_table_name='test_table'

time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

ckpt_model_name='s2_ndcg-cp-p-n_ckpt_gte-base_5layer_plus_2024-02-21_21-22-33'
ckpt_model_index=3
ckpt_model_dir=os.path.join('model',ckpt_model_name)

config_dict={'batch_size':candidate_size,'ckpt_model_name':ckpt_model_name,'ckpt_model_index':ckpt_model_index,'time_data':time_data}

test_link_table=os.path.join('dataset_new',test_link_table_name+'.npy')
file_dir=os.path.join('dataset_new','text')
test_data=MAGDataset_cpVSp(file_dir,test_link_table)
test_loader=DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

indices=np.load(os.path.join(ckpt_model_dir,f'test_indices_{ckpt_model_index}-candidate{candidate_size}.npy'))


os.environ["OPENAI_API_KEY"]= 'YOUR_API_KEY'

client = OpenAI()

indices_record=list()
test_prec_cp_5=list()

batch_count=0
f_cnt=0

with torch.no_grad():
    for batch in tqdm(test_loader):
        response_file=os.path.join('code','result','json',f"{batch_count}.json")
        time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        response_json=[]

        paper_q,paper_cp1,paper_cp2,paper_cp3,paper_cp4,paper_cp5,paper_p1,paper_p2,paper_p3,paper_p4,paper_p5=batch

        paper_k=paper_cp1+paper_cp2+paper_cp3+paper_cp4+paper_cp5+paper_p1+paper_p2+paper_p3+paper_p4+paper_p5

        paper_indices=indices[batch_count*batch_size:batch_count*batch_size+batch_size]

        new_index_list=list()
        for i in tqdm(range(batch_size)):
            title,abstract=paper_q[i].split('\n')
            prompt=analyzer_example

            prompt+='\n<question>\n'

            prompt+="Here is the title and abstract of the query paper.\n"
            prompt+=f"Title: {title}\nAbstract: {abstract}\n"
            
            prompt+='Here are some other research papers which have been already cited by the query paper.\n'
            for t,paper in enumerate([paper_k[j] for j in paper_indices[i][2:8]]):
                title,abstract=paper.split('\n')
                prompt+=f"{t}\nTitle: {title}\nAbstract: {abstract}\n"
            
            prompt+='Try to think abductively and convince yourself as a researcher. Figure out why the query paper cite these one by one.'
            prompt+='Try to think step by step before giving the answer.</question>'

            print(prompt)
            completion = client.chat.completions.create(

            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            n=1,
            )
            analysis=completion.choices[0].message.content
            usage=completion.usage
            analyzer_completion_tokens=usage.completion_tokens
            analyzer_prompt_tokens=usage.prompt_tokens
            analyzer_total_tokens=usage.total_tokens

            rank_system_pr='Your role is to assist in predicting which research papers are most likely to be cited together based on a given set of papers or topics. Strive for fairness and objectivity.'
            title,abstract=paper_q[i].split('\n')

            prompt=ranker_example
            prompt+='\n<question>\n'
            prompt+="Here is the title and abstract of the query paper.\n"
            prompt+=f"Title: {title}\nAbstract: {abstract}\n"
            prompt+='There are some other candidate papers and the analysis of why this query paper cites these.\n'
            prompt+=f"<analysis>\n{analysis}\n</analysis>"
            prompt+=f'\nNow you are doing a research following up this query paper ({title}) above. '
            prompt+='\nUse the analysis to identify patterns or themes that suggest potential citation relationships.'
            prompt+='\nRank these candidate papers in the order you are most likely to cite from the perspective of a research follower and provide explanations or justifications for your reasoning.'
            prompt+='\nAfter finishing your analysis with ranking result, please answer an extra single line in the end following the format {ranked order: paper 0/1/2/3/4/5, paper 0/1/2/3/4/5, paper 0/1/2/3/4/5, paper 0/1/2/3/4/5, paper 0/1/2/3/4/5, paper 0/1/2/3/4/5} (one number each) and do not include any other words.</question>'
           
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": rank_system_pr},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            n=1,
            )

            rank=completion.choices[0].message.content
            usage=completion.usage
            ranker_completion_tokens=usage.completion_tokens
            ranker_prompt_tokens=usage.prompt_tokens
            ranker_total_tokens=usage.total_tokens

            str_num=re.sub(r'\D', '', rank)
            if len(str_num)<6:
                str_num="012345"
                f_cnt+=1

            add_index1=int(str_num[-6])+2
            add_index2=int(str_num[-5])+2
            add_index3=int(str_num[-4])+2
            add_index4=int(str_num[-3])+2
            add_index5=int(str_num[-2])+2
            add_index6=int(str_num[-1])+2
            add_index=[add_index1,add_index2,add_index3,add_index4,add_index5,add_index6]
            index_set=set(add_index)
            if index_set != {2,3,4,5,6,7}:
                f_cnt+=1
                add_index=[2,3,4,5,6,7]
                add_index1,add_index2,add_index3,add_index4,add_index5,add_index6=[2,3,4,5,6,7]

            update_rank=np.concatenate(([paper_indices[i][add_index1]],[paper_indices[i][add_index2]],[paper_indices[i][add_index3]],[paper_indices[i][add_index4]],[paper_indices[i][add_index5]],[paper_indices[i][add_index6]]))
            paper_indices[i][2:8]=update_rank
            p_index_list=paper_indices[i].tolist()
            response_json.append({
                'batch_count':batch_count,
                'paper_index':i,
                'reranking_index':p_index_list,
                'analysis':analysis,
                'analyzer_completion_tokens':analyzer_completion_tokens,
                'analzer_prompt_tokens':analyzer_prompt_tokens,
                'analyzer_total_tokens':analyzer_total_tokens,
                'rank':rank,
                'ranker_completion_tokens':ranker_completion_tokens,
                'ranker_prompt_tokens':ranker_prompt_tokens,
                'ranker_total_tokens':ranker_total_tokens,
                'total_tokens':analyzer_total_tokens+ranker_total_tokens
                })

        #print(f_cnt)    
        with open(response_file, "a+") as json_file:
            json.dump(response_json, json_file, indent=4,cls=npencoder)

        batch_count+=1
