## TRADE Multi-Domain and Unseen-Domain Dialogue State Tracking

This is my baseline implementation of the paper:

**Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems**. [**Chien-Sheng Wu**](https://jasonwu0731.github.io/), Andrea Madotto, Ehsan Hosseini-Asl, Caiming Xiong, Richard Socher and Pascale Fung. **_ACL 2019_**.
[[PDF]](https://arxiv.org/abs/1905.08743)

To run these scripts, you will need the following libraries

- torch
- numpy
- tqdm
- embeddings
- matplotlib

To download the MultiWOZ dataset and process it for DST

```shell
python3 create_data.py
```

To train a simple model that matches performance in original paper

```shell
python3 train.py --log_path=log.json
```

To test the best model, find the encoder/decoder models in /save/TRADE-multiwozDST and select the model with highest dev set accuracy
Model names follow the pattern HDD400-BSZ4-DR0.2-ACC-0.4867

- HDD = embedding dimension
- BSZ = batch size
- DR = dropout percent
- ACC = development set accuracy

For this given model, we would test by

```shell
MODEL_PATH=save/TRADE-multiwozDST/HDD400-BSZ4-DR0.2-ACC-0.4867
python3 test.py --model_path=$MODEL_PATH --log_path=log.json
```


To run the scripts with NER, you will need to install spacy, as well as a pretrained NER model
```shell
pip install spacy
python -m spacy download en_core_web_sm
```