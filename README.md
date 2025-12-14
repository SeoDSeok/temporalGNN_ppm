# temporalGNN_ppm
Multi-Task Learning for Predictive Process Monitoring with Temporal GNN on OCEL

## Data
We have obtained each original data below.
- bpi 2014 : https://data.4tu.nl/datasets/7d097cec-7304-4b85-9e78-a3ca1cc44c40/1
- bpi 2016 : https://data.4tu.nl/datasets/95613fb2-29a5-49dc-b196-0948cf96cd7c/1
- bpi 2017 : https://data.4tu.nl/datasets/6889ca3f-97cf-459a-b630-3b0b0d8664b5/1
- bpi 2019 : https://data.4tu.nl/datasets/46a7e15b-10c7-4ab2-988d-ee67d8ea515a/1

## Data preprocessing
- bpi 2014 : data_preprocessing_2014.py
- bpi 2016 : data_preprocessing_2016.py
- bpi 2017 : data_preprocessing.py (main)
- bpi 2019 : data_preprocessing_2019.py

Then, we get this folder structure
```
- tgn_input_af_per_case
    - case_Application_95215
        - edges.csv
        - node_features.npy
        - nodes.csv
    - case_Application_220112
    - case_Application_235300
    ...
```
Uploaded data is just sample data from bpi challenge 2017 (ocel)

## Model
We basically conducted the experiment using TGAT and compared it with the TGN model used in ICPM 2025. Below is an additional comparison model.

- Simple Model
    - MLP
- Sequence Model
    - LSTM
    - GRU
- Graph-based Model
    - GGCN
    - TGN
- Transformer Model
    - Process Transformer

And we also conducted an ablation study.

- w/o time encoding
- w/o Context time
- w/o Context
- w/o Multi Head
- w/o Activity embedding

## Experiments Results
Please refer to our paper
