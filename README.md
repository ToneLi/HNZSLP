## A  Hierarchical N-Gram Framework for Zero-Shot Link Prediction [EMNLP Findings 2022]
This is the source code of [HNZSLP](https://arxiv.org/abs/2204.10293). For the full project, including NELL-ZS, Wiki-ZS datasets and progressed relational surface name, please refer to [Google Drive](https://drive.google.com/file/d/1Mro57n-F9P552qW5jPVDQZkdbzOTp26L/view?usp=sharing).


### 1) Overview

The architecture of our proposed HNZSLP is depicted in the diagram below.   It consists of three major parts:

<img src="https://github.com/ToneLi/HNZSLP/blob/main/framework.jpg" width="500"/>

**Part 1**: Hierarchical N-gram Graph Constructor constructs a hierarchical n-gram graph from each relation surface name, where the n-gram graph can be further decomposed into an {\it adjoin} graph and {\it compositional} graph to simplify the learning of the relation representation; 

**Part 2**:  GramTransformer constructs the relation representation by modeling over the adjoin and compositional graphs;

**Part 3**:  KG Embedding Learning Module combines the embeddings of the head entity and relation to predict the tail entity and update the embeddings.


### 2) Configuration

Python 3.6.6

torch: 1.5.1

Linux: CentOS Linux release

### 3)Run Code

```markdown
 For excample: please enter: LZSL\LZSP_nell_Distmult_13_90
 You just run "CUDA_VISIBLE_DEVICES=1 python main.py" in your terminal.
 Please use the default parameters.
```


### 4)Core code for the baseline model (Byyt5)
#### Byt5 core code

[Byt5(2021)](https://github.com/google-research/byt5),
[Byt5(hugging face)](https://huggingface.co/docs/transformers/model_doc/byt5)
```
#transformers version:4.7.0
from transformers import T5ForConditionalGeneration, AutoTokenizer
byte_model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

enc = tokenizer(["Life is like a box of chocolates.", "Today is Monday.","Today is Monday."], padding="longest", return_tensors="pt")

M = byte_model(input_ids=enc["input_ids"].cuda(),
                attention_mask=enc["attention_mask"].cuda(),
                decoder_input_ids=enc["input_ids"].cuda())  # .encoder_last_hidden_state # forward pass

relation_text_embedding=M.logits
r = relation_text_embedding.sum(1)

"""
transformers---version 4.7.0
tf---version 1.6.0
torch version: 1.5.1
"""
```
#### Chartransformer  core code
[CharFormer(2021)](https://github.com/google-research/google-research/tree/master/charformer),
(CharFromer (implenment)) (https://github.com/lucidrains/charformer-pytorch)
```
    self.tokenizer = GBST(
            num_tokens=26, # I just use 26 letter (a,b,.....)
           
            dim=200,  # dimension of token and intra-block positional embedding
            max_block_size=4,  # maximum block size
            downsample_factor=1,  # the final downsample factor by which the sequence length will decrease by
            score_consensus_attn=True
            # whether to do the cheap score consensus (aka attention) as in eq. 5 in the paper
        )
        
        # r_idx: [[1,2,3,4,5],[4,5,6,7,8],[7,8,9,06,7],....]  len(r_idx)==batch size, such as 64
        ids_r = torch.tensor(r_idx[j]).unsqueeze(0).cuda()
        relations_feather= self.tokenizer(ids_r)[0]
        relations_feather = self.postion_embedding(relations_feather)
        relations_feather = self.encoder(relations_feather)
        relations_feather = self.mean_pooling(relations_feather)
        
torch: pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
charformer-pytorch--charformer-pytorch-0.0.4
python 3.6
```

### 5) How to get the  surfacename of wiki relation 

#### Go to the web [wiki query](https://query.wikidata.org/)
input:
```
SELECT ?property ?propertyLabel ?propertyDescription (GROUP_CONCAT(DISTINCT(?altLabel); separator = ", ") AS ?altLabel_list) WHERE {
    ?property a wikibase:Property .
    OPTIONAL { ?property skos:altLabel ?altLabel . FILTER (lang(?altLabel) = "en") }
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" .}
 }
GROUP BY ?property ?propertyLabel ?propertyDescription
LIMIT 5000
```

####  you will got a file query.json, use the following code to get the relation and its surfacename 

```
import json
with open("query.json",'r',encoding="utf-8") as load_f:
     load_dict = json.load(load_f)
# DIC=load_dict
ids=[]
surface_name=[]
for r in load_dict:
    ids.append(r["property"].replace("http://www.wikidata.org/entity/",""))
    surface_name.append(r["propertyLabel"])
DIC=dict(zip(ids,surface_name))
# print(DIC)
fw=open("relation_surface_name_id.txt","w",encoding="utf-8")
i=0
with open("relation_to_description.txt","r",encoding="utf-8") as fr:
    for line in fr.readlines():
        r=line.strip().split("\t")[0]
        if r in DIC:
            fw.write(r+"\t"+DIC[r]+"\n")
        else:
            i=i+1
            print(r)
print(i)

```

### 6) Sentence Representation by Word2vec and TF-IDF


```
import numpy as np
from sklearn import feature_extraction
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import  CountVectorizer
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

i=0
corpus=[]
with open("WSP_Q_set.txt","r") as fr:
    for line in fr.readlines():
        i=i+1

        text_=line.strip().replace("[","").replace("]","")
        corpus.append(text_)
        # sentence_embeddings = model.encode([text_])
        # R.append(sentence_embeddings[0])

    #
    # R=np.array(R)
    # np.save("metaqa_hop1_vector.npy",R)
# corpus=["good boy yes","girl project"]
vectorizer=CountVectorizer()
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
word=vectorizer.get_feature_names()
R=[]
weight=tfidf.toarray()

for i in range(len(weight)):
    print(i)
    vector_al=[]
    for j in range(len(word)):
        # print(word[j], weight[i][j])

        w=word[j]
        w_w=weight[i][j]
        if w_w !=0 and w in model:
            vector=model[w]
            vector_al.append(vector)


    V=np.sum(np.array(vector_al),axis=0)
    R.append(V)
    # print(V)
    # print("----------")


np.save("WSP_vector.npy",R)
```


