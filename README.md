Our BERT-based keyphrase extraction system is implemented based on an open source project **[NER-BERT-pytorch](https://github.com/lemonhu/NER-BERT-pytorch)**

## Data Source

The ScienceIE corpus can be downloaded at:
https://scienceie.github.io/resources.html

The ACL keyphrase corpus can be downloaded at:
https://nlp.stanford.edu/pubs/FTDDataset_v1.txt

## Usage

- **Get BERT model for PyTorch**

  You can download the pre-trained model named scibert from https://github.com/allenai/scibert。

- **Train your own model**

  >  bash run.sh

- **Evaluate this model**

  > bash evaluate.sh

## reference

Augenstein I, Das M, Riedel S, Vikraman L, McCallum A. Semeval 2017 task 10:
Scienceie-extracting keyphrases and relations from scientific publications. In:
Proceedings of the 11th International Workshop on Semantic Evaluation; 2017. p.
546-555.

Gupta S, Manning C. Analyzing the dynamics of research by extracting key
aspects of scientific papers. In: Proceedings of 5th international joint conference
on natural language processing; 2011. p. 1-9.

Devlin J, Chang MW, Lee K, Toutanova K. BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding. In: Proceedings of the
2019 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, Volume 1 (Long and
Short Papers); 2019. p. 4171-4186.
