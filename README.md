## CharNet

A multi-layer perceptron network for artificial text and character generation

### General

CharNet is a [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
network using using a dense feed-forward-network in which every layer is connected with
all previous layers. Therefore CharNet is an application of the principles behind 
[GAU](https://github.com/ClashLuke/GAU) and
[InDeDeNet](https://github.com/ClashLuke/InDeDeNet). \
It acts as a humanly-readable example to show improvements compared to Transformers,
RNNs and other state-of-the-art models for sequential data.\
Various parameters are supported, with most of them being switches, allowing for high
configurability. Note that all parameters already have tested and well-behaving default
values, so huge amounts of tweaking are not necessary.\

### Getting Started

If you're interested in running this code on one of
[Google's free cloud GPUs](colab.research.google.com/), you could follow the notebook in 
[here](https://colab.research.google.com/github/ClashLuke/CharNet/blob/master/CharNet_Notebook.ipynb).

If you're a hard-core user who wants to run it on their own machine, you should start
with cloning this repository.
``` 
$ git clone https://github.com/ClashLuke/CharNet
```
Afterwords a python file or interpreter can be opened to first import the CharNet
interface and then run the training.
```PYTHON
from CharNet import CharNet
network = CharNet()
network.run(datasetFilePath='dataset.dataset',
            prepare_text=False)
keras_model = network.model
```
As depicted above, the production-ready Keras model can be extracted after training 
by accessing the .model attribute of the CharNet instance.  

### Toy Datasets
* A **3GB** example dataset created out of all books found on the
[full books site](http://www.fullbooks.com/) can be downloaded in
[here](https://drive.google.com/file/d/1oBe5jVnk9PrOIitnD2B02-8fSQyxxT0R/view?usp=sharing).
It was formatted to use a
[minimalistic character set](https://github.com/ClashLuke/CharNet/blob/master/mlp/utils.py#L30).
* A **500MB** dataset based on the data from
[textfiles](http://www.textfiles.com/directory.html) can be downloaded in
[here](https://drive.google.com/file/d/1e4NZNhKqZCqzapnDgqYdEM02gzx81ZsW/view?usp=sharing).
While using the same character set as the dataset above, it is still significantly more
noisy.\
* A third, significantly smaller dataset, is a dataset containing all tweets by Donald
Trump, as seen in [here](http://www.trumptwitterarchive.com/archive). Its only 
**5MB**, contains links and did not undergo any special formatting. It can be found in
[here](https://drive.google.com/file/d/1GifcAh7D2puKgu2k4oapmrDGetdoWfFC/view?usp=sharing).\
* Lastly there also is a **616MB** dump of the linux kernel with removed comments which
can be found [here](https://drive.google.com/open?id=1bAVryLcuL0k-BjNZHMinueoznkiZIzUD).
This dataset is pure as well, allowing for the true code generation experience.
* For stress-testers, there also is the [PG-19](https://github.com/deepmind/pg19) 
dataset by deepmind. It contains **11GB** of pure, unformatted books in multiple
languages. The model has been tested using this dataset.

Note that for most of these datasets, it is possible to attach them to your own google
drive. You can then use them in google's colaboratory, as mentioned in
[Getting Started](#Getting-Started).
It is recommended to copy those datasets to your google drive to then use them in [Google Colab](https://colab.research.google.com/) if you are looking to try this project out.

### Parameters

| Parameter name    | Description                                                                                                                                                                                                                                                                     | Datatype | Default value |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ------------- |
| **leakyReLU**     | [LeakyReLU](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L19) is a rectifier function which can be used as an activation layer. See [this](https://miro.medium.com/max/1025/1*ypsvQH7kvtI2BhzR2eT_Sw.png) graphic for more information. | Boolean     | False         |
| **batchNorm**     | [Batch Normalization](https://arxiv.org/abs/1502.03167) is a technique that can be [used in keras](https://github.com/keras-team/keras/blob/master/keras/layers/normalization.py#L16) to achieve better results by normalizing the activations of the previous layer.           | Boolean     | False         |
| **trainNewModel** | Specifies whether a new model should be trained or an old one should be loaded to continue training.                                                                                                                                                                            | Boolean     | True          |
|**repeatInput**| Append the input layer to the input of every layer. | Boolean | True|
|**unroll**| When using a LSTM on a non-GPU device, LSTMs can be unrolled to reduce computational cost while increasing RAM cost drastically.| Boolean | True|
|**splitInputs**| If a large number of inputs is used, the inputs can be split to many perceptrons which all receive only part of the input.| Boolean | False|
|**initialLSTM**| Use an LSTM as the first layer after the input layer. LSTMs tend to have a better performance for sequence prediction, yet are against the idea of an MLP.| Boolean | False|
|**inputDense**| Adds a smaller layer in front of the hidden layers to reduce calculation. The number of neurons in it is equal to the number of inputs.| Boolean | False|
|**splitLayer**| Similar to splitInputs, layers can be split as well when they receive many inputs. This particular implementation also allows data to be fed from left to right of the model. It is highly discoured to use this parameter as it disables every form of parallelization. | Boolean | False|
|**concatDense**| Concatenate all previous layers to one big input layer for the next hidden layer. See [model.png](https://github.com/ClashLuke/CharNet/blob/master/model.png) for more information. This is the main point of research of CharNet. | Boolean | True |
|**bidirectional**| If a LSTM is used, it can either be [bidirectional](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks) or not. Bidirectional LSTMs tend to understand constructs better but also require twice the time to train and evaluate. | Boolean | True |
|**concatBeforeOutput**| If activated, all previous layers will be concatenated to one large layer to be fed into the output layer. See [model.png](https://github.com/ClashLuke/CharNet/blob/master/model.png) for more information. | Boolean | True |
|**drawModel**| Draws a graphic as seen in [model.png](https://github.com/ClashLuke/CharNet/blob/master/model.png) using keras' built-in functions to help visualize the networks architecture. | Boolean | True |
|**gpu**| Specifies whether a NVIDIA GPU is used or not. Using an NVIDIA GPU allows for massive optimizations in LSTM calculation. | Boolean | True |
|**indexIn**| If activated, indexes or labels are used as inputs instead of one-hot encoded labels. For more information, see [this](https://towardsdatascience.com/choosing-the-right-encoding-method-label-vs-onehot-encoder-a4434493149b) blog post. | Boolean | False |
|**classNeurons**| To improve the user experience, a parameter that automatically multiplies the number of neurons provided with the number of classes used is added. If used, a hidden layer will have significantly more neurons allowing for higher performance when using one-hot encoding. | Boolean | True |
|**inputs**| Specifies the number of characters used as an input for the neural network. | Integer | 60 |
|**neuronsPerLayer**| Specifies the number neurons in each hidden layer. In most cases 2*inputs is sufficient. | Integer | 120 |
|**layerCount**| The total number of _hidden_ layers used in the network. Generally four is the highest recommended value. | Integer | 4 |
|**epochs**| Number of real epochs where the entire dataset was passed through the neural network. | Integer | 1 |
|**kerasEpochsPerEpoch**| To receive frequent updates on large datasets, one can set this parameter so that keras calls the callbacks more often. Warning: This does interfere with the training optimizer, potentially causing negative results. | Integer | 256 |
|**learningRate**| The rate the neural network learns with. The most important hyperparameter to test and tweak. | Float | 0.005 |
|**outputs**| Number of outputs in characters. It is highly discouraged to use more than one output. | Integer | 1|
|**dropout**| [Dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) is a regularization technique to make sure the network learns instead of remembering trainings data. Values from 0.15 to 0.4 can be used without significantly harming the network.| Integer | 0.35 |
|**batchSize**| The batch size is the number of training samples the neural network uses to make _one_ update to the weights. A higher number improves the speed and reduces overfitting but might also harm the network. | Integer | 1024 |
|**valSplit**| Specifies how much of the input dataset should be used for validation at the end of every keras epoch. Generally 100MB or more should be set aside for validation. | Float | 0.1 |
|**verbose**| Used to set a verbosity level in keras. According to keras' documentation: `0 = silent, 1 = progress bar, 2 = one line per epoch.` | Integer | 1 |
|**outCharCount**| Number of characters to be recursively generated using a test string at every end of a keras epoch.| Integer | 512 |
|**changePerKerasEpoch**| Sets the percentage of the original batch size that should be added to the current batch size after every keras epoch. Example: `Epoch 1: Batch Size 100; Epoch 1: Batch Size 200; Epoch 3: Batch Size 300`  | Float | 0.25 |
|**activation**|Activation function used in the neural network. The default is [gelu](https://arxiv.org/pdf/1606.08415.pdf) but [other activation functions](https://keras.io/activations/) can be used as well. | String | 'gelu'|
|**weightFolderName**|Defines the name of the folder weights are saved to and loaded from.|String|'MLP_Weights'|
|**testString**|A string that is used as an input for the neural network to predict at the end of every keras epoch. If set to None, it will default to [this](https://github.com/ClashLuke/CharNet/blob/master/mlp/utils.py#L62) string. |String|None|
|**charSet**|A set of characters used in the text used as input. Leaving at None will make the network assume it is [this](https://github.com/ClashLuke/CharNet/blob/master/mlp/utils.py#L30) char set. A text can be formatted to this format by passing `prepare_text=True` to your charnet instance or explicitly calling `charnet.prepare_text()`. |String|None|

### Todo

- [x] Explain parameters
- [x] Make code humanly readable
- [x] Add config dict instead of enforcing parameters to be assigned
- [x] Create Notebooks to link to
- [x] Add link to example datasets
- [ ] Add proper example showing most interface functions using the trump dataset as an example.
- [ ] Add example output
