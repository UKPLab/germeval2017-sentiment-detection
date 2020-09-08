# Germeval2017 Code for Task C and D

This is a modifed version of the MTL Sequence Tagging Framework by [Tobias Khase](https://github.com/UKPLab/thesis2018-tk_mtl_sequence_tagging).
The changes include:

* Upgraded to Python 3.6 
* Upgraded to Tensorflow 2.x

# Running the experiments

The following provides a simple guideline to re-run the experiments with 100 dimensional word embeddings in our paper. 

## Setup

Note: 
* To run ```./fetch_data.sh``` you may have to create the folder ```postag_word2vec``` in ```data/embeddings/```.
* The embeddings you can download on [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2483) have been filtered to match the vocabulary of the dataset (to save storage space).  

1. Download the data and embeddings:  

    ```./fetch_data.sh```

1. Set up an appropriate environment  

    ```conda create --name <env-name> python=3.6```
    
    ```conda activate <env-name>```

    ```pip install -r requirements.txt```


## Configuration

Configuration files can be found in ```experiment_configurations```. They are YAML files that specify the experimental configurations. For more details please check [CONFIGURATION.md](./CONFIGURATION.md).


## Training

You can train a model via:

```python main.py train <path-to-config>```

```main.py``` is located in ```/src```. For example, to run the experiment on task D using postagged embeddings do:

```cd /src```

```python main.py train ../experiment_configurations/germeval_taskD_iob_postags.yaml```

That will generate a model in the ```/pkl``` folder and save the outputs to ```/out```.


## Evaluating models

You can evaluate models via:

```python main.py eval <path-to-model> <path-to-config> ```






