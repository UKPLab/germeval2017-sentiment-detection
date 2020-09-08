# Network Configuration

The network can be configured via configuration files in [YAML](http://www.yaml.org/) format.
These configuration files can be quite complex and thus the different configuration options will be
explained in this document.

## Configuration Parts

The configuration consists of multiple parts:
* [Experiment configuration](#experiment-configuration)
* [Task configuration](#task-configuration)
* [File configuration](#file-configuration)
* [Hidden layer configuration](#hidden-layer-configuration)
* [Early stopping configuration](#early-stopping-configuration)
* [Character level information configuration](#character-level-information-configuration)
* [Embedding configuration](#embedding-configuration)
* [Training configuration](#training-configuration)

All parts will be explained in the following sections.

> NOTE: not specifying a configuration option is equivalent to setting its value to `None`.

### Experiment Configuration

| Configuration Option                  | Description | Supported Values | Default Value |
| ------------------------------------- | ----------- | ---------------- | ------------- |
| name                                  | Name of the experiment. This value is also used to name the output folder for log files, predictions, models, etc. | Any string that can be used as a directory name. | `"my_experiment"` |
| num_runs                              | This option allows to specify how often this network is trained and evaluated. Evaluation results will be averaged across all runs in the end. | Any positive integer above 0. Usually values between 5 and 10. | `10` |
| epochs                                | Define how many epochs are performed to train the network. | Any positive integer above 0. | `30` |
| batch_size                            | Maximum mini-batch size used during training. The actual mini-batch size depends on the sentence length distribution in the data set. | Any positive integer above 0. | `32` |
| use_variational_dropout               | Whether to use variational dropout (repeat the same dropout mask for each time step) or not. See [Gal and Ghahramani (2016)](https://arxiv.org/pdf/1512.05287.pdf). | `True` or `False` | `True` |
| short_cut_connections                 | Whether to use short-cut connections (feed the word representation, e.g. embeddings, into each shared layer) or not. See [Hashimoto et al. (2017)](https://arxiv.org/pdf/1611.01587.pdf) | `True` or `False` | `False` |
| tasks                                 | A list of task configurations. See [Task Configuration](#task-configuration) for further details. | A YAML list of task configurations. | `[]` |
| early_stopping                        | An early stopping configuration object. See [Early Stopping Configuration](#early-stopping-configuration) for further details. | An early stopping configuration or `None` to disable early stopping. | `None` |
| character_level_information           | A character level information configuration object. See [Character Level Information Configuration](#character-level-information-configuration) for further details. | A character level information configuration object or `None` to disable using character level information. | `None`.
| eval_metrics                          | A list of evaluation metric names. These metrics are used by all tasks. NOTE: you can also specify task-specific evaluation metrics. | <ul><li>`"accuracy"`</li><li>`"f1"`</li><li>`"f1_o"`</li><li>`"f1_b"`</li><li>`"precision"`</li><li>`"precision_o"`</li><li>`"precision_b"`</li><li>`"recall"`</li><li>`"recall_o"`</li><li>`"recall_b"`</li><li>`"am_components_0.5"`</li><li>`"am_components_0.999"`</li><li>`"am_relations_0.5"`</li><li>`"am_relations_0.999"`</li><li>`"word_accuracy"`</li><li>`"avg_edit_distance"`</li><li>`"median_edit_distance"`</li></ul> | `"accuracy"`, `"f1"`, `"precision"`, and `"recall"` |               
| rnn_unit                              | Which type of RNN cell to use. | `"simple"`, `"GRU"`, and `"LSTM"` | `"LSTM"` |
| rnn_dropout_input_keep_probability    | Keep probability for the input of RNN cells. Dropout = 1.0 - keep probability. | A floating point value in `[0.0, 1.0]`. `1.0` is equivalent to not using dropout at all. | `1.0` |
| rnn_dropout_output_keep_probability   | Keep probability for the output of RNN cells. Dropout = 1.0 - keep probability. | A floating point value in `[0.0, 1.0]`. `1.0` is equivalent to not using dropout at all. | `1.0` |
| rnn_dropout_state_keep_probability    | Keep probability for the state of RNN cells. Dropout = 1.0 - keep probability. | A floating point value in `[0.0, 1.0]`. `1.0` is equivalent to not using dropout at all. | `1.0` |
| use_bias                              | Whether or not to use bias in the shared layers. | `True` or `False` | `True` |
| units                                 | Number of units in the shared (RNN) layers for one direction. The shared layer is always a bi-directional RNN. | Any positive integer above 0. | `100` |
| word_dropout_keep_probability         | Keep probability for input words. Dropout = 1.0 - keep probability. | A floating point value in `[0.0, 1.0]`. `1.0` is equivalent to not using dropout at all. | `1.0` |
| embeddings                            | A list of embedding configurations. See [Embedding Configuration](#embedding-configuration) for further details. | A list of embedding configurations or `None` to disable using pre-trained embeddings. | `None` |
| embedding_size                        | Dimensionality of the word embeddings. This option is only used if no pre-trained word embeddings have been specified. | Any positive integer above 0. | `100` |
| training                              | A list of training configurations. See [Training Configuration](#training-configuration) | A training configuration object or `None` to use the default settings (see [Training Configuration](#training-configuration) for default parameters). | `None` |


### Task Configuration

| Configuration Option      | Description | Supported Values | Default Value |
| ------------------------- | ----------- | ---------------- | ------------- |
| name                      | Name of the task. This name is used to reference the task, e.g. in [early stopping](#early-stopping-configuration) and to name result and prediction files that contain task-specific information. | Any string without characters that cannot be used in file names, e.g. `:`. | `"Task_i"` where `i` is the index of the task in the task list. |
| train_file                | A file configuration object. See [File Configuration](#file-configuration) for further details. | A file configuration object. | No default. |
| dev_file                  | A file configuration object. See [File Configuration](#file-configuration) for further details. | A file configuration object or `None`. | `None` |
| test_file                 | A file configuration object. See [File Configuration](#file-configuration) for further details. | A file configuration object or `None`. | `None` |
| output_layer              | Index (i.e. `0` refers to the first shared layer) of the shared layer that is used to feed the task-specific hidden layers and classifier. | Any positive integer starting with 0. | `1` |
| hidden_layers             | A list of hidden layer configurations. See [Hidden Layer Configuration](#hidden-layer-configuration) for further details. The hidden layers are placed between the shared layer and the task classifier. | A list of hidden layer configurations or `None` to disable using task-specific hidden layers (except for the projection layer that is required for the classifier. | `None` |
| ~~loss~~                  ||||
| ~~loss_weight~~           ||||
| eval_metrics              | A list of task-specific evaluation metrics. These metrics are only used for this task. This is useful for metrics that cannot be applied to any task, i.e. `"am_components_0.5"`. | <ul><li>`"accuracy"`</li><li>`"f1"`</li><li>`"f1_o"`</li><li>`"f1_b"`</li><li>`"precision"`</li><li>`"precision_o"`</li><li>`"precision_b"`</li><li>`"recall"`</li><li>`"recall_o"`</li><li>`"recall_b"`</li><li>`"am_components_0.5"`</li><li>`"am_components_0.999"`</li><li>`"am_relations_0.5"`</li><li>`"am_relations_0.999"`</li><li>`"word_accuracy"`</li><li>`"avg_edit_distance"`</li><li>`"median_edit_distance"`</li></ul> If this option is `None` no task-specific metrics are used. | `None` |
| classifier                | Which classifier to use for the task. | `"softmax"` or `"CRF"` | "`softmax`" |
| data_format               | The format of the data files used in this task. This option is used to select the appropriate data reader. | `"CONLL"` | `"CONLL"` |
| dropout_keep_probability  | Keep probability for the classifier input. Dropout = 1.0 - keep probability. | A floating point value in `[0.0, 1.0]`. `1.0` is equivalent to not using dropout at all. | `1.0` |
| use_bias                  | Whether or not to use bias in the projection layer. | `True` or `False` | `True` |
| encoding                  | The encoding used in the data files used in this task. This option is used to select appropriate post-processing methods. | `"NONE"`, `"BIO"`, `"IOB"`, and `"IOBES"` | `"NONE"` |
| type                      | The type of the task. This option is used to select appropriate post-processing methods. | `"GENERIC"` and `"AM"` (for argumentation mining) | `"GENERIC"` | 

### File Configuration

| Configuration Option      | Description | Supported Values | Default Value |
| ------------------------- | ----------- | ---------------- | ------------- |
| path                      | Path to the file. If this is a relative path, the path is always relative to `$PWD` when executing `python main.py` or `python run_experiment.py`. | Any valid path. | No default. |
| column_separator          | Which character is used to separate the columns in the file. NOTE: it is assumed that the file is in a column-based format, e.g. CoNLL. | `"space"` and `"tab"` | `"tab"` |
| word_column               | Index (i.e. `0` refers to the first column) of the column that contains the word/token. | Any positive integer starting with 0. | `0` |
| label_column              | Index (i.e. `0` refers to the first column) of the column that contains the label. | Any positive integer starting with 0. | `1` |
| encoding                  | The encoding of the file. | Any value that is supported by Python's [codecs.open](https://docs.python.org/2/library/codecs.html#codecs.open) | `"utf8"` |

### Hidden Layer Configuration
| Configuration Option      | Description | Supported Values | Default Value |
| ------------------------- | ----------- | ---------------- | ------------- |
| units                     | Number of units in the hidden layer. | Any positive integer above 0. | `100` |
| activation                | Which activation function to use in the hidden layer. | `"tanh"`, `"linear"`, `"relu"`, and `"sigmoid"` | `"relu"` |
| dropout_keep_probability  | Keep probability for the output of the hidden layer. Dropout = 1.0 - keep probability. | A floating point value in `[0.0, 1.0]`. `1.0` is equivalent to not using dropout at all. | `1.0` |
| use_bias                  | Whether or not to use bias in the hidden layer. | `True` or `False` | `True` |

### Early Stopping Configuration

> NOTE: the early stopping configuration is also required to store the best model. Otherwise, the latest model is always stored. If you want to store the best model, but without early stopping, set `patience` to a value that is equal
or greater than the number of epochs specified in the [Experiment Configuration](#experiment-configuration)

| Configuration Option      | Description | Supported Values | Default Value |
| ------------------------- | ----------- | ---------------- | ------------- |
| task_name                 | Name of a task from the task list. | Any task name that occurs in the task list. | `"task_name"` |
| metric                    | Name of the metric that is used to decide whether the model improved or not. | <ul><li>`"accuracy"`</li><li>`"f1"`</li><li>`"f1_o"`</li><li>`"f1_b"`</li><li>`"precision"`</li><li>`"precision_o"`</li><li>`"precision_b"`</li><li>`"recall"`</li><li>`"recall_o"`</li><li>`"recall_b"`</li><li>`"am_components_0.5"`</li><li>`"am_components_0.999"`</li><li>`"am_relations_0.5"`</li><li>`"am_relations_0.999"`</li><li>`"word_accuracy"`</li><li>`"avg_edit_distance"`</li><li>`"median_edit_distance"`</li></ul> | `"f1"` | 
| patience                  | Number of epochs to wait before stopping although the performance did not improve. | Any positive integer starting with 0. | `5` |


### Character Level Information Configuration
| Configuration Option      | Description | Supported Values | Default Value |
| ------------------------- | ----------- | ---------------- | ------------- |
| network_type              | Type of network used to obtain character level information. | `"LSTM"` (see [Lample et al. (2016)](https://arxiv.org/pdf/1603.01360.pdf)) and `"CNN"` (see [Ma and Hovy (2016)](https://arxiv.org/pdf/1603.01354.pdf); not supported right now) | `"LSTM"` |
| dimensionality            | Dimensionality of the character embeddings resulting from the character level information extractor. | Any positive integer above 0. | `100` | 
| hidden_units              | Number of hidden units in the extractor. Only used for the LSTM extractor. | Any positive integer above 0. | `100` | 


### Embedding Configuration
| Configuration Option      | Description | Supported Values | Default Value |
| ------------------------- | ----------- | ---------------- | ------------- |
| path                      | Path to the embeddings file. If this is a relative path, the path is always relative to `$PWD` when executing `python main.py` or `python run_experiment.py`. | Any valid path. | No default. |
| encoding                  | The encoding of the file. | Any value that is supported by Python's [codecs.open](https://docs.python.org/2/library/codecs.html#codecs.open) | `"utf8"` |
| size                      | Dimensionality of the embeddings. Used to verify that the embeddings file was read correctly. | The number of dimensions in the embeddings file. | No default. |
| gzip                      | Whether or not the embeddings file is gzipped. | `True` or `False` | `True` |


### Training Configuration
| Configuration Option      | Description | Supported Values | Default Value |
| ------------------------- | ----------- | ---------------- | ------------- |
| optimizer                 | Which optimizer to use. | `"SGD"`, `"adam"`, `"adagrad"`, `"adadelta"` | `"adam"` |
| optimizer_params          | See the [Tensorflow documentation](https://www.tensorflow.org/api_guides/python/train#optimizers) for further information. The object defined for this property is passed directly to the optimizer initialization. NOTE: some optimizers have required parameters that **must** be specified here, e.g. `"sgd"` ([tf.train.GradientDescentOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)) requires the specification of the learning rate. | `{}` |
| use_gradient_clipping     | Whether to use gradient clipping (using clipping by the global norm; see [tf.clip_by_global_norm](https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm)). | `True` or `False` | `True` |
| clip_norm                 | The clipping ratio used for gradient clipping. This setting is ignored if `use_gradient_clipping` is set to `False`. | Any positive floating point value. | `5.0` |

### Example

```yaml
name: POS_and_Chunk
num_runs: 10
epochs: 50
batch_size: 32
optimizer: adam
use_variational_dropout: True
short_cut_connections: True
tasks:
  - name: POS
    train_file:
      path: ../data/wsj/pos/wsj_pos_15-18.train
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    dev_file:
      path: ../data/wsj/pos/wsj_pos.dev
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    test_file:
      path: ../data/wsj/pos/wsj_pos.test
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    output_layer: 0
    eval_metrics: []
    classifier: softmax
    data_format: CONLL
    dropout_keep_probability: 0.8
    use_bias: True
    encoding: NONE
    type: GENERIC
  - name: Chunk
    train_file:
      path: ../data/wsj/chunk/wsj_chunk_15-18.train
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    dev_file:
      path: ../data/wsj/chunk/wsj_chunk.dev
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    test_file:
      path: ../data/wsj/chunk/wsj_chunk.test
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    output_layer: 1
    hidden_layers:
      - units: 100
        activation: relu
        dropout_keep_probability: 1.0
        use_bias: True
    eval_metrics:
      - f1
      - precision
      - recall
    classifier: CRF
    data_format: CONLL
    dropout_keep_probability: 0.8
    use_bias: True
    encoding: BIO
    type: GENERIC 
early_stopping:
  task_name: Chunk
  metric: f1
  patience: 5
character_level_information:
  network_type: LSTM
  dimensionality: 100
  hidden_units: 100
eval_metrics:
  - accuracy
rnn_unit: LSTM
rnn_dropout_input_keep_probability: 0.8
rnn_dropout_output_keep_probability: 0.8
rnn_dropout_state_keep_probability: 0.8
use_bias: True
units: 100
word_dropout_keep_probability: 0.9
embeddings:
    - path: ../data/embeddings/wiki_extvec_words_gz.gz
      encoding: utf8
      size: 300
      gzip: True
```

## Random Search Templates
When performing a random search, the [ConfigGenerator](src/ConfigGenerator.py) uses a template file to generate multiple
configuration trials. The trials are generated from multiple intervals that are also specified in the template file.

The template file is in YAML format and contains two YAML documents (separated by three dashes, `---`). The first YAML
document contains the intervals and the second document holds the actual template.

Each interval has a name and the template contains the names of the intervals as variables. After sampling a trial from
the intervals, the variables are replaced by the actual values from the trial.

For further information see the following files which are all involved in running random searches:

* [TrialGenerator](src/TrialGenerator.py)
* [ConfigGenerator](src/ConfigGenerator.py)
* [run_experiment](src/run_experiment.py)

In the following, an example of a template is shown. In this example, there is only an interval that chooses the word
dropout keep probability from the interval `[0.5, 1.0]`.

```yaml
WORD_DROPOUT_KEEP_PROBABILITY:
  interval_type: continuous
  start: 0.5
  end: 1.0
---
name: POS_and_Chunk
num_runs: 10
epochs: 50
batch_size: 32
optimizer: adam
use_variational_dropout: True
short_cut_connections: True
tasks:
  - name: POS
    train_file:
      path: ../data/wsj/pos/wsj_pos_15-18.train
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    dev_file:
      path: ../data/wsj/pos/wsj_pos.dev
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    test_file:
      path: ../data/wsj/pos/wsj_pos.test
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    output_layer: 0
    eval_metrics: []
    classifier: softmax
    data_format: CONLL
    dropout_keep_probability: 0.8
    use_bias: True
    encoding: NONE
    type: GENERIC
  - name: Chunk
    train_file:
      path: ../data/wsj/chunk/wsj_chunk_15-18.train
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    dev_file:
      path: ../data/wsj/chunk/wsj_chunk.dev
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    test_file:
      path: ../data/wsj/chunk/wsj_chunk.test
      column_separator: space
      word_column: 0
      label_column: 1
      encoding: utf8
    output_layer: 1
    hidden_layers:
      - units: 100
        activation: relu
        dropout_keep_probability: 1.0
        use_bias: True
    eval_metrics:
      - f1
      - precision
      - recall
    classifier: CRF
    data_format: CONLL
    dropout_keep_probability: 0.8
    use_bias: True
    encoding: BIO
    type: GENERIC 
early_stopping:
  task_name: Chunk
  metric: f1
  patience: 5
character_level_information:
  network_type: LSTM
  dimensionality: 100
  hidden_units: 100
eval_metrics:
  - accuracy
rnn_unit: LSTM
rnn_dropout_input_keep_probability: 0.8
rnn_dropout_output_keep_probability: 0.8
rnn_dropout_state_keep_probability: 0.8
use_bias: True
units: 100
word_dropout_keep_probability: WORD_DROPOUT_KEEP_PROBABILITY
embeddings:
    - path: ../data/embeddings/wiki_extvec_words_gz.gz
      encoding: utf8
      size: 300
      gzip: True
```
