# Question Answering Using Memory Networks
The code implements End-to-End Learning on bAbI Question Answering tasks using Memory Networks. Facebook's MemN2N Architecture [1] has been implemented as a skeleton model, with additional modifications that have been mentioned below:

* LSTM for Sentence Representation
* Shared Embedding Matrix for Question and Context
* Temporal Encoding
* Linear Start
* Training Embedding Matrices using Skipgram

The final modified architecture which we used can be seen here: https://go.gliffy.com/go/share/image/slwgpn5126xykk3vnjav.png

For generating the sentence representation in embedding space, using word vector embeddings, **LSTM** was used, in place of simple matrix multiplication in MemN2N. LSTM captures the structure of sentence, which is useful for many tasks that have a huge dependency on sentence structure and word positions.

To ensure that dot products between question and context take place in same embedding space, **Shared Embedding Matrices** were used, i.e. same embedding matrix was used for projecting the context and question to embedding space.

Further to emphasize the order of sentences in which they appear in memory, **Temporal Encoding** was used after transformation to embedding space.

To avoid vanishing gradient in the early stages of training, **Linear Start** was added, in which the softmax layers were initially removed, except the last softmax and tranining commenced. When the validation loss started to saturate, those softmax layes were re-inserted.

Instead of training embedding matrices in an end-to-end fashion, they were pre-trained using Skipgram method. Facebook's `fastText`[2] library was used for this task. These trained embeddings were then used directly in the model, in a shared embedding fashion.

## Libraries Used
* NumPy
* Pytorch (torch, torchvision)

Additional basic libraries were used as well, like `sys`, `os`, `defaultdict`.

## Implementation
### Data Input and Pre-Processing
The `get_data` function in **data_transform.py** is used to get the vocabulary and transformation of training, testing and validation data in Bag of Words and Positional Encoding Representation.

```
get_data(train_fname, valid_fname, test_fname, vec_fname='bAbI_Data/model.vec', unk_thres=0, pre_embed = False)
```

* **train\_fname** : [mandatory] The txt file name of training data.

* **valid\_fname** : [mandatory] The txt file name of validation data.

* **test\_fname** : [mandatory] The txt file name of test data.

* **vec\_fname** : The file name of pre-trained vector embeddings. Required if *pre_embed* = True.

* **pre\_embed** : [optional] Flag indicating whether to use pre-trained embedding matrices or not. Default value = False

Additionally, if *pre_embed* = True, the function `get_embeddings` has to be invoked which returns the embedding matrix as a numpy array.

### Model Definition
The `QuesAnsModel` class in **model_funcs.py** has details of the model. While creating a new model, following arguments can be passed in the constructor of the class:

```
__init__(self,embedding_dim, vocab_size, num_hops = 1, max_mem_size=15,temporal=False,same=0,positional=False,dropout=0,pre_embed=False,embed_wts=None)
```

* **embedding\_dim** : [mandatory] The dimension of embedding space has to be passed here. This will be overwritten if *pre\_embed* = True.

* **vocab\_size** : [mandatory] The number of words in vocabulary.

* **num\_hops** : [optional] The number of hops for the model forward pass. Default value = 1

* **max\_mem\_size** : [optional] The maximum number of most recent sentences to be stored in memory. Default value = 15

* **temporal** : [optional] Flag indicating whether to use Temporal Encoding or not. Default value = False

* **same** : [optional] Flag indicating whether to use Shared Embedding Matrix or not. Default value = 0

* **positional** : [optional] Flag indicating whether to use LSTM for sentence representation or not. Default value = False

* **dropout** : [optional] Dropout to be added for LSTM. Used only when *positional* = True. Default value = 0

* **pre\_embed** : [optional] Flag indicating whether to use pre-trained embedding matrices or not. Default value = False

* **embed\_wts** : The embedding weight matrix to be used. Required if *pre\_embed* = True. Default value = None

### Train Function
The function `train` in **model_funcs.py** is used to train the model:

```
train(model,tr_dt_bow,vd_dt_bow,tr_dt_pe, vd_dt_pe,opt=optim.Adam,epochs=10,eta=0.0001,LS=0,ls_thres=0.001,model_name ='a_model_has_no_name',visualize=True)
```

* **model** : [mandatory] The model to train.

* **tr\_dt\_bow** : [mandatory] Training data in Bag of Words Representation.

* **vd\_dt\_bow** : [mandatory] Validation data in Bag of Words Representation.

* **tr\_dt\_pe** : [mandatory] Training data in Positional Encoding Representation.

* **vd\_dt\_pe** : [mandatory] Validation data in Positional Encoding Representation.

* **opt** : [optional] Optimizer to be used. Default value = `optim.Adam`

* **epochs** : [optional] Number of epochs for which to run the model. Default value = 10

* **eta** : [optional] Learning Rate to be used. Default value = 0.0001

* **LS** : [optional] Flag indicating whether to use Lienar Start or not. Default value = 0. Value 1 implies using Linear Start. Value -1 implies using only Linear layer and no softmax ever.

* **ls\_thres** : Saturation threshold for linear start. Required if *LS* = 1. Default value = 0.001

* **model\_name** : [optional] The model name with which you want the figures and model to be stored. Default value = 'a\_model\_has\_no\_name'

* **visualize** : [optional] Flag indicating whether to visualize model progress with increasing epochs or not. Default value = True

### Test Function
The function `test` in **model_funcs.py** is used to test the model on test data.

```
test(model,test_dt_bow,test_dt_pe)
```

* **model** : [mandatory] The model to be tested.

* **test\_dt\_bow** : [mandatory] Test data in Bag of Words Representation.

* **test\_dt\_pe** : [mandatory] Test data in Positional Encoding Representation.

### Visualization Function
The function `test_visualize` in **model_funcs.py** is used to visualize probability graphs during training and testing.

```
test_visualize(model,test_dt_bow,test_dt_pe, get_probs=True, num_words)
```

* **model** : [mandatory] The model to visualize.

* **test\_dt\_bow** : [mandatory] Test data in Bag of Words Representation.

* **test\_dt\_pe** : [mandatory] Test data in Positional Encoding Representation.

* **get\_probs** : [mandatory] Flag indicating whether to display probabilities along with prediction or not. Default value = True

* **num\_words** : Number of words with highest probabilities to be displayed. Required if *get\_probs* = True

### Driver Code
The `Skeleton_Code.ipynb` jupyter file is the driver code for this task. Once the flags mentioned above for different functions are set, this code takes care of everything for executing the program successfully.

### Additional .py Files
Three additional files are provided:

* **model_funcs_GPU.py** : This has GPU dependencies for the code.

* **model_funcs_pt2.py** : This is to be used if the machine has Pytorch2 installed instead of Pytorch3.

* **model_funcs_pt2_GPU.py** : This is to be used with Pytorch2 and GPU.

The driver code handles this as well, once the flags `GPU` and `pyTorch2` flags are set appropriately.

## References
[1] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, "End-to-End Memory Networks", arXiv:1503.08895v5 [cs.NE]

[2] https://github.com/facebookresearch/fastText/
