# Code for Machine Learning Experiments

This folder includes the code for machine learning experiments.

## Folder structure

### vocab.py
Contains the **Vocab** class for conversion between words (strings) and their 
computational representations (indices of characters). The exemplar of this 
class initialized with all Russian alphabetic symbols is used in the whole 
pipeline.

### models.py
Contains the architectures for PyTorch models and the corresponding utils.
Includes:
- **LSTMIsCompoundClassifier**;
- **LSTMHypothesesClassifier**.

### is_compound_training.py
A script that 
- trains a compound-or-not classification model, 
- computes validation and test metrics,
- selects and saves the best checkpoint according to a validation F1 score, and
- shows the examples of false positive and false negative samples from the test 
  set. 

### hypotheses_scorer_training.py
TODO: Hermann