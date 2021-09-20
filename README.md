# Compound Splitting and Analysis for Russian

This repository contains the code and the data for the paper `Compound Splitting and Analysis for Russian`.

## Quick Start

1. Clone the repository.
2. Install the requirements: ```pip install -r requirements.txt```.
Make sure that you have Python 3.6+.
   TODO: add requirements from DerivBase.Ru (or add setup.py to DerivBase.Ru)
3. Run ```python /code/data_manip/gold_analyses_splitter.py```.
4. Run ```./code/data_manip/download.sh```.
5. Run ```python /code/ml/is_compound_training.py``` to train the is-compound 
   classifier.
6. TODO. Clone DerivBase.Ru, ..., run ...
7. Run ... to evaluate baselines for hypotheses scoring.
8. Run ... to evaluate the ML model for hypotheses scoring.

## License

TODO

## Citing
If you are using this work in your research, please, do not forget to cite it.

The corresponding paper was accepted to the DeriMo 2021 workshop and will appear in its proceedings.

### Publications

- Daniil Vodolazsky and Hermann Petrov. Compound Splitting and Analysis for Russian (DeriMo 2021).

@article{vodolazskycompound,
  title={Compound Splitting and Analysis for Russian},
  author={Vodolazsky, Daniil and Petrov, Hermann}
}

- Daniil Vodolazsky. DerivBase.Ru: a Derivational Morphology Resource for Russian (LREC 2020).

```
@InProceedings{vodolazsky:2020:LREC,
author = {Vodolazsky, Daniil},
title = {DerivBase.Ru: a Derivational Morphology Resource for Russian},
booktitle = {Proceedings of The 12th Language Resources and Evaluation Conference},
month = {May},
year = {2020},
address = {Marseille, France},
publisher = {European Language Resources Association},
pages = {3930--3936},
abstract = {Russian morphology has been studied for decades, but there is still no large high coverage resource that contains the derivational families (groups of words that share the same root) of Russian words. The number of words used in different areas of the language grows rapidly, thus the human-made dictionaries published long time ago cannot cover the neologisms and the domain-specific lexicons. To fill such resource gap, we have developed a rule-based framework for deriving words and we applied it to build a derivational morphology resource named DerivBase.Ru, which we introduce in this paper.},
url = {https://www.aclweb.org/anthology/2020.lrec-1.484}
}
```