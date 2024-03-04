# KHSCP
This repository contains the code and dataset for our paper "Leveraging Pre-existing Resources for Data-Efficient Counter-Narrative Generation in Korean".

## Overview

We propose a **Korean Hate Speech Counter Punch (KHSCP)**, a cost-effective counter-narrative generation recipe in the Korean language. To this end, we construct the first hate speech counter-narrative dataset in Korean. To enhance counter-narrative generation performance, we propose an effective augmentation method and investigate the reasonability of a large-scale language model to overcome data scarcity in low-resource environments by leveraging existing resources. 



## Dataset

### Source Dataset
Our dataset is constructed through translation based on the [multitarget CONAN dataset](https://aclanthology.org/2021.acl-long.250/). The categories of hate speech include MUSLIMS, MIGRANTS, WOMEN, LGBT+, JEWS, POC, OTHER, and DISABLED, and are composed of pairs of hate speech and counter-narratives. For the training of generative models, there are 4,002 for training, 500 for validation, and 501 for the test set respectively.

### Pre-existing Resources
We utilize four Korean hate speech resources to augment Korean counter-narrative pairs through Semantic-based Based Augmentation (SBA). Each dataset is publicly available and must be downloaded to execute the code, to be stored in /data/mono_hs_data.
  - [APEACH](https://aclanthology.org/2022.findings-emnlp.525/)
  - [BEEP](https://aclanthology.org/2020.socialnlp-1.4/)
  - [Unsmile](https://arxiv.org/abs/2204.03262)
  - [KOLD](https://aclanthology.org/2022.emnlp-main.744/)

The required directory structure in our repository is as follows.

```bash
📦kor_cn
 ┣ 📂data
 ┃ ┣ 📂integrated
 ┃ ┣ 📂matched
 ┃ ┣ 📂mono_hs_data
 ┃ ┃ ┣ 📂apeach
 ┃ ┃ ┃ ┗ 📜test.csv
 ┃ ┃ ┣ 📂beep
 ┃ ┃ ┃ ┣ 📜beep_full.csv
 ┃ ┃ ┃ ┣ 📜train.tsv
 ┃ ┃ ┃ ┗ 📜dev.tsv
 ┃ ┃ ┣ 📂kold
 ┃ ┃ ┃ ┗ 📜kold_v1.json
 ┃ ┃ ┗ 📂unsmile
 ┃ ┃ ┃ ┣ 📜unsmile_train_v1.0.tsv
 ┃ ┃ ┃ ┣ 📜unsmile_valid_v1.0.tsv
 ┃ ┃ ┃ ┗ 📜unsmile_full.csv
 ┃ ┣ 📂threshold_integrated
 ┃ ┣ 📜train.csv
 ┃ ┣ 📜valid.csv
 ┃ ┗ 📜test.csv
 ┣ 📂results
 ┣ 📜dataloader.py
 ┣ 📜generate.py
 ┣ 📜integrate.py
 ┣ 📜rouge_score.py
 ┣ 📜semantic_match.py 
 ┣ 📜train.py
 ┗ 📜utils.py
 ┣ 📜run_generation_bart.sh
 ┣ 📜run_generation_gpt.sh
 ┣ 📜run_generation_t5.sh
 ┣ 📜run_generation_threshold.sh
 ┣ 📜run_train_bart.sh
 ┣ 📜run_train_gpt.sh
 ┣ 📜run_train_t5.sh
 ┣ 📜run_train_threshold.sh
```


## How to run
In the following section, we describe how to train and evaluate each model by using our code.

### Semantic Matching
For each instance of existing hate speech, the closest counter-narrative is matched by model-based similarity searching on hate speech and is saved along with the similarity score.
```python
python semantic_match.py --data_path ./data/mono_hs_data/apeach/test.csv
python semantic_match.py --data_path ./data/mono_hs_data/unsmile/beep_full.csv
python semantic_match.py --data_path ./data/mono_hs_data/unsmile/unsmile_full.csv
python semantic_match.py --data_path ./data/mono_hs_data/unsmile/kold_v1.json
```

### Integration
For the matched pairs, only pairs that have values above a certain threshold are extracted. They are combined with the existing training set to form an augmented set.
```python
python integrate.py --threshold 0.6
```

### Training
We support models such as mT5, BART, and GPT2 during the learning process, and it is possible to modify detailed options through argument adjustments within the shell script.
```bash
sh run_train_gpt.sh
sh run_train_bart.sh
sh run_train_t5.sh
```

To see changes according to each threshold value, run the shell below.
```bash
sh run_train_threshold.sh
```

### Evaluation
Evaluate the generation performance of the trained model. The metric consists of ROUGE and BLEU scores based on the Korean Mecab tokenizer.
```bash
sh run_generation_gpt.sh
sh run_generation_bart.sh
sh run_generation_t5.sh
```

To see changes according to each threshold value, run the shell below.
```bash
sh run_generation_threshold.sh
```


## Reference
```bibtex
@inproceedings{fanton-2021-human,
  title="{Human-in-the-Loop for Data Collection: a Multi-Target Counter Narrative Dataset to Fight Online Hate Speech}",
  author="{Fanton, Margherita and Bonaldi, Helena and Tekiroğlu, Serra Sinem and Guerini, Marco}",
  booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics",
  month = aug,
  year = "2021",
  publisher = "Association for Computational Linguistics",
}
```
```bibtex
@inproceedings{yang-etal-2022-apeach,
    title = "{APEACH}: Attacking Pejorative Expressions with Analysis on Crowd-Generated Hate Speech Evaluation Datasets",
    author = "Yang, Kichang  and
      Jang, Wonjun  and
      Cho, Won Ik",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.525",
    pages = "7076--7086",
    abstract = "In hate speech detection, developing training and evaluation datasets across various domains is the critical issue. Whereas, major approaches crawl social media texts and hire crowd-workers to annotate the data. Following this convention often restricts the scope of pejorative expressions to a single domain lacking generalization. Sometimes domain overlap between training corpus and evaluation set overestimate the prediction performance when pretraining language models on low-data language. To alleviate these problems in Korean, we propose APEACH that asks unspecified users to generate hate speech examples followed by minimal post-labeling. We find that APEACH can collect useful datasets that are less sensitive to the lexical overlaps between the pretraining corpus and the evaluation set, thereby properly measuring the model performance.",
}
```
```bibtex
@inproceedings{moon-etal-2020-beep,
    title = "{BEEP}! {K}orean Corpus of Online News Comments for Toxic Speech Detection",
    author = "Moon, Jihyung  and
      Cho, Won Ik  and
      Lee, Junbum",
    booktitle = "Proceedings of the Eighth International Workshop on Natural Language Processing for Social Media",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.socialnlp-1.4",
    pages = "25--31",
    abstract = "Toxic comments in online platforms are an unavoidable social issue under the cloak of anonymity. Hate speech detection has been actively done for languages such as English, German, or Italian, where manually labeled corpus has been released. In this work, we first present 9.4K manually labeled entertainment news comments for identifying Korean toxic speech, collected from a widely used online news platform in Korea. The comments are annotated regarding social bias and hate speech since both aspects are correlated. The inter-annotator agreement Krippendorff{'}s alpha score is 0.492 and 0.496, respectively. We provide benchmarks using CharCNN, BiLSTM, and BERT, where BERT achieves the highest score on all tasks. The models generally display better performance on bias identification, since the hate speech detection is a more subjective issue. Additionally, when BERT is trained with bias label for hate speech detection, the prediction score increases, implying that bias and hate are intertwined. We make our dataset publicly available and open competitions with the corpus and benchmarks.",
}
```
```bibtex
@misc{kang2022korean,
    title={Korean Online Hate Speech Dataset for Multilabel Classification: How Can Social Science Aid Developing Better Hate Speech Dataset?},
    author={TaeYoung Kang and Eunrang Kwon and Junbum Lee and Youngeun Nam and Junmo Song and JeongKyu Suh},
    year={2022},
    eprint={2204.03262},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
```bibtex
@inproceedings{jeong2022kold,
  title={KOLD: Korean Offensive Language Dataset},
  author={Jeong, Younghoon and Oh, Juhyun and Lee, Jongwon and Ahn, Jaimeen and Moon, Jihyung and Park, Sungjoon and Oh, Alice Haeyun},
  booktitle={The 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022},
  year={2022},
  organization={EMNLP}
}
```
