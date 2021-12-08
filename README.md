# LatticeTransformer-for-Zero-shot-learning
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
||  Byt5(2021)  |    |   |   | |-    |-  |-  |-
||  transformer (char input)  |    |   |   | |-    |-  |-  |-
||  CharFormer(2021)  |    |   |   | |-    |-  |-  |-
|DistMult(relation-surface)|  BERT  |    |   |   | |-    |-  |-  |-
||  Byt5(2021)  |    |   |   | |-    |-  |-  |-
||  CharFormer(2021)  |    |   |   | |-    |-  |-  |-
|ComplEx(relation-surface)|  BERT  |    |   |   | |-    |-  |-  |-
||  Byt5(2021)  |    |   |   | |-    |-  |-  |-
||  CharFormer(2021)  |    |   |   | |-    |-  |-  |-

[Byt5(2021)](https://github.com/google-research/byt5)

[CharFormer(2021)](https://github.com/google-research/google-research/tree/master/charformer)

