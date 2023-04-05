## Active learning selection methods used in "On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model: An Empirical Study on Codex"

```angular2html
@article{zhuo2023robustness,
  title={On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model: An Empirical Study on Codex},
  author={Zhuo, Terry Yue and Li, Zhuang and Huang, Yujin and Li, Yuan-Fang and Wang, Weiqing and Haffari, Gholamreza and Shiri, Fatemeh},
  journal={arXiv preprint arXiv:2301.12868},
  year={2023}
}
```

### Requirements
```
conda create -n AL-SP python=3.8
conda activate AL-SP
conda install editdistance
conda install nltk
conda install scikit-learn
pip install torch
conda install -c huggingface transformers
pip install scikit-learn-extra
pip install argparse
```

### Usage
```
python sample_methods.py --sample_size 50 --sample_method nl_LM_feature --dataset_path data/train.tsv --outputdir savedir
``` 
