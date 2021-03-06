# 1. iDAF

## 1.1. Table of contents

- [1. iDAF](#1-idaf)
  - [1.1. Table of contents](#11-table-of-contents)
  - [1.2. General](#12-general)
  - [1.3. Getting Started](#13-getting-started)
  - [1.4. Toy Datasets](#14-toy-datasets)
    - [1.4.1. Description](#141-description)
    - [1.4.2. List](#142-list)
  - [1.5. Parameters](#15-parameters)
  - [1.6. Technical](#16-technical)
    - [1.6.1. Structure](#161-structure)
    - [1.6.2. RNN](#162-rnn)
    - [1.6.3. Transformer](#163-transformer)

## 1.2. General

The insanely deep attention fabric (iDAF) is a [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) network creating a dense fabric of attention blocks, with each block being connected with all previous blocks.\
It acts as a humanly-readable example to show improvements compared to Transformers, RNNs and other state-of-the-art models for sequential data.\
Various parameters are supported, allowing for high configurability. Note that all parameters already have tested and well-behaving default values, so huge amounts of tweaking are not necessary.

## 1.3. Getting Started

If you're interested in running this code on one of Google's free cloud GPUs, you could follow the notebook in [Colab](https://colab.research.google.com/github/ClashLuke/CharNet/blob/master/CharNet.ipynb)
or [Kaggle](https://www.kaggle.com/alfodawin/charnet). Note that Colab appears to have some issues with IPython right now, rendering it unusable. An example with extreme parameters can be found [on kaggle](https://www.kaggle.com/alfodawin/charnet?scriptVersionId=32268469) as well. It aims to illustrate that the model can converge and generalize well, even
with high batch sizes and huge potential to overfit.

If you're a hard-core user who wants to run it on their own machine, you should start with cloning this repository.

```BASH
$ git clone https://github.com/ClashLuke/iDAF
Cloning into 'iDAF'...
remote: Enumerating objects: 68, done.
remote: Counting objects: 100% (68/68), done.
remote: Compressing objects: 100% (44/44), done.
remote: Total 763 (delta 47), reused 44 (delta 24), pack-reused 695
Receiving objects: 100% (763/763), 813.67 KiB | 355.00 KiB/s, done.
Resolving deltas: 100% (512/512), done.
```

Afterwords a python file or interpreter can be opened to first import the iDAF
interface and then run the training.

```PYTHON
from iDAF import iDAF
network = iDAF()
network.train('iDAF/tinyshakespeare.txt')
keras_model = network.model
```

As depicted above, the production-ready Keras model can be extracted after training by accessing the .model attribute of the iDAF instance.  

## 1.4. Toy Datasets

### 1.4.1. Description

To remove the hassle of having to download a dataset, this repository comes with the "tinyshakespeare" dataset out of the box. It's **1MB** of shakespeare's works. Bigger datasets are available through google drive or google buckets. \
If you want to use any of the datasets below, you could either download them to your local machine or copy them to your drive and use them in colab. There are utility scripts which can be used to mount and copy the datasets from your google drive to your colab instance.

### 1.4.2. List

- A **3GB** example dataset created out of all books found on the
[full books site](http://www.fullbooks.com/) can be downloaded in
[here](https://drive.google.com/file/d/1oBe5jVnk9PrOIitnD2B02-8fSQyxxT0R/view?usp=sharing). It was formatted to use a [minimalistic character set](https://github.com/ClashLuke/CharNet/blob/master/mlp/utils.py#L30).
- A **500MB** dataset based on the data from [textfiles](http://www.textfiles.com/directory.html) can be downloaded in [here](https://drive.google.com/file/d/1e4NZNhKqZCqzapnDgqYdEM02gzx81ZsW/view?usp=sharing). While using the same character set as the dataset above, it is still significantly more noisy.
- A third, significantly smaller dataset, is a dataset containing all tweets by Donald Trump, as seen in [here](http://www.trumptwitterarchive.com/archive). Its only **5MB**, contains links and did not undergo any special formatting. It can be found in [here](https://drive.google.com/file/d/1GifcAh7D2puKgu2k4oapmrDGetdoWfFC/view?usp=sharing).
- Lastly there also is a **616MB** dump of the linux kernel with removed comments which can be found [here](https://drive.google.com/open?id=1bAVryLcuL0k-BjNZHMinueoznkiZIzUD). This dataset is pure as well, allowing for the true code generation experience.
- For stress-testers, there also is the [PG-19](https://github.com/deepmind/pg19) dataset by deepmind. It contains **11GB** of pure, unformatted books in multiple languages. The model has been tested using this dataset.


## 1.5. Parameters

| Parameter            | Description                                                                                                           | Datatype | Default                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------- |
| neurons_per_layer    | Number of neurons (features) used in every layer of the neural network. Only used if neuron_list is not given.        | Int      | 16                                |
| layer_count          | Number of layers (blocks) for the entire network. Only used if neuron_list is not given.                              | Int      | 4                                 |
| inputs               | Number of previous instances (characters) used to predict the next.                                                   | Int      | 16                                |
| classes              | Number of classes the input is mapped to. Ideally a value close to the real number of unique instances.               | Int      | 30                                |
| dropout              | Amount of noise applied between blocks. Between 0 and 1.                                                              | Float    | 0.3                               |
| input_dropout        | Amount of noise applied on the input. Between 0 and 1.                                                                | Float    | 0.1                               |
| batch_size           | Number of examples the model sees at once before making an update to its parameters.                                  | Int      | 1024                              |
| learning_rate        | Amount of parameter update done after seeing one batch of examples. Bigger batch size enables bigger learning rates.  | Float    | 1e-3                              |
| generated_characters | Number of characters to generate when one training epoch ends. Can be set to 0.                                       | Int      | 512                               |
| neuron_list          | List of feature counts for every block of the network. Overwrites neurons_per_layer and layer_count.                  | List     | []                                |
| block_depth          | List containing the number of residual layers used to make up a block.                                                | List     | []                                |
| metrics              | List of metrics used to track the performance of the model.                                                           | List     | ['accuracy']                      |
| embedding            | Whether to use the raw input data or instead take the input as indices of a generated a matrix. Improves performance. | Bool     | True                              |
| class_neurons        | Whether to feed class-based data or numbers through the network.                                                      | Bool     | True                              |
| load_model           | Whether to load the latest model written to the model_folder from disk instead of creating a new one.                 | Bool     | False                             |
| output_activation    | Activation function applied to the output. None (without quotes) means no activation, allowing linear regression.     | Str      | "softmax"                         |
| loss                 | Error function the model tries to optimize.                                                                           | Str      | "sparse_categorical_crossentropy" |
| model_folder         | Folder trained model snapshots get saved to.                                                                          | Str      | "mlp_weights"                     |

## 1.6. Technical

### 1.6.1. Structure

The insanely deep attention fabric is model-space building on top of the [insanely deep dense network](https://github.com/ClashLuke/InDeDeNet) architecture, while adding key features from the gateless variant of the [gated attention unit](https://github.com/ClashLuke/GAU).\
It consists of attention-blocks, shaped similarly to those of a transformer. However, the value-layer takes the output of the key-layer as an input, allowing for more transformations with a lower number of blocks.\
Additionally, the number of dropout layers, normalization and activation layes is reduced to the bare minimum. This improves both the execution time and overall model performance.\
Other than that, the model itself is not optimized for speed. For a potentially faster PyTorch implementation, visit the [GAU](https://github.com/ClashLuke/GAU). Do note however that both tensorflows static graph execution and XLA do a lot of optimization, rendering the improvements of a much more pipelined modul useless.

### 1.6.2. RNN

As the RNN is a building block, but the iDAF is a complete architecture, we will instead compare the iDAF to a sequence-to-sequence model using nothing but RNNs.\
Compared to those, the iDAF has similar advantages as a transformer. The iDAF relies on parallelizable and pipelinable attention blocks, which themselves contain nothing but highly optimized, cheap matrix multiplications.\
When comparing RNNs with transformers however, one immediately notices the extreme difference in depth. Each cell of an RNN is its own (recurrent) layer, meaning that an RNN usually has a hundred times more layers than a transformer.\
The iDAF attempts to bridge this difference, by creating insanely deep attention fabrics.

### 1.6.3. Transformer

As mentioned above, the iDAF is significantly deeper than a transformer could be. Additionally, the iDAF uses a DenseNet architecture for its blocks. where each blocks outputs are connected to all previous outputs.
This allows the iDAF to save on resources while reducing its abilities to overfit, as it doesn't require huge numbers of features, but instead relies mostly on depth.\
[Google](https://ai.googleblog.com/2019/10/exploring-massively-multilingual.html) recently discovered that increasing the depth of a model reduces generally improves performance much more than increasing the width of the model would. They found that using wider models leads to a higher chance of memorization and therefore overfitting, which isn't ideal for language modeling.\
The iDAF leverages those findings, to further improve upon the existing transformer model.
