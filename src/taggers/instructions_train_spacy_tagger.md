# Instructions for training custom transformer-based / Tok2Vec tagger using spaCy.


Everything will be done in the command line.


<!-- 1. First make sure that the text data is in the spacy binary format. If not, convert it using the [`convert`](https://spacy.io/api/cli#convert) command: -->
<!-- 
```bash
python -m spacy convert /home/rsaha/projects/babylm/data/train_50M_multimodal_clean/ /home/rsaha/projects/babylm/data/spacy_bin_data/ --converter conllu -C true
``` -->
