# LatticeTransformer-for-Zero-shot-learning  (This is a formidable discover--by chen)
LatticeTransformer (word, bytes,..) for Zero shot learning

## (Task 1) Link prediction in Zero Shot Learning 

![1636244513(1)](https://user-images.githubusercontent.com/33856565/140627681-720760e2-29dc-40c4-b37f-8427366f9729.jpg)

On the top, ZSL-(KGE model) is the baseline is proposed in ZSGAN, such as ZSL-Tucker. (ZSGAN, OntoZSL) are two models from 2020, 2021. LENR is the model we proposed.

**Baseline group 1:ZSGAN, OntoZSL **

**Baseline group 2:About relation surfacename**

(--------NELL-ZS-----------Wiki-ZS---------)
| KGE  | Model     | MRR     | hits@10     |hits@5     |hits@1|MRR     |hits@10     |hits@5     |hits@1|
|---------- | ---------- | :-----------:  | :-----------: |:-----------: |:-----------: | :-----------: |:-----------: |:-----------: |:-----------: |
|TransE(relation-surface)|  BERT  |  0.269  |  0.402 | 0.343  |0.196 |-    |-  |-  |-
||  Byt5(2021)  |   0.103 |0.173   |0.129   |0.064 |-    |-  |-  |-
||  transformer (word input)  |    |   |   | |-    |-  |-  |-
||  transformer (char input)  | 0.285   |  0.411 |  0.349 |0.217 |-    |-  |-  |-
||  CharFormer(2021)  |  0.265  | 0.391  |0.338   | 0.191|-    |-  |-  |-
|DistMult(relation-surface)|  BERT  |    |   |   | |-    |-  |-  |-
||  Byt5(2021)  |    |   |   | |-    |-  |-  |-
||  CharFormer(2021)  |    |   |   | |-    |-  |-  |-
|ComplEx(relation-surface)|  BERT  |    |   |   | |-    |-  |-  |-
||  Byt5(2021)  |    |   |   | |-    |-  |-  |-
||  CharFormer(2021)  |    |   |   | |-    |-  |-  |-

[Byt5(2021)](https://github.com/google-research/byt5),
[Byt5(hugging face)](https://huggingface.co/docs/transformers/model_doc/byt5)

[CharFormer(2021)](https://github.com/google-research/google-research/tree/master/charformer),
(CharFromer (implenment)) (https://github.com/lucidrains/charformer-pytorch)

#### Byt5 core code
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
