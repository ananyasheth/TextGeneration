### Text Generation using LSTM and SimpleRNN

#### Introduction

In this project, we explore the use of Recurrent Neural Networks (RNNs) for text generation using the Plato corpus as the source text. The two types of RNNs we use are Long Short-Term Memory (LSTM) and Simple Recurrent Neural Network (SimpleRNN). A probabilistic language model is created to generate new text that is similar in style and content to the original text.

#### Mathematical Foundation

At the heart of the probabilistic language model is the concept of a conditional probability. Given a sequence of words, the goal is to predict the next word in the sequence. This can be formalized as finding the probability of a word $w_t$ given its context, represented as the sequence of previous words $(w_1, w_2, ..., w_{t-1})$.

$$P(w_t | w_1, w_2, ..., w_{t-1}) = \frac{count(w_1, w_2, ..., w_{t-1}, w_t)}{count(w_1, w_2, ..., w_{t-1})}$$

The probabilistic language model is trained to maximize the likelihood of the observed data. This can be done by minimizing the negative log likelihood, which is commonly used as a loss function in RNNs.

$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P(w_{i,t} | w_{i,1}, w_{i,2}, ..., w_{i,t-1})$$

Where $N$ is the number of sequences in the training data, $T_i$ is the length of the $i^{th}$ sequence, and $w_{i,t}$ is the $t^{th}$ word in the $i^{th}$ sequence.

#### Model Architecture

Embedding Layer:
The first layer of the model is an Embedding layer. The purpose of this layer is to map each word in the vocabulary to a dense vector of fixed size. The vocab_size is the size of the vocabulary, which is the number of unique words in the corpus, and chronology_len is the length of the chronology of words for each input sample.

LSTM Layers:
The next two layers are Long Short-Term Memory (LSTM) layers. LSTMs are a type of Recurrent Neural Network (RNN) that are designed to handle the issue of vanishing gradients, which is a problem that arises when training traditional RNNs for longer sequences. The first LSTM layer has return_sequences=True, which means it will return the hidden state output for each input time step. The second LSTM layer has return_sequences=False, which means it will return the hidden state output for the final time step only.

Dense Layers:
The next four layers are Dense layers. Dense layers are fully connected neural network layers, which means each neuron in a dense layer is connected to every neuron in the previous layer. The activation function used in these layers is the Rectified Linear Unit (ReLU), which is defined as f(x) = max(0, x).

Dropout Layers:
The two Dropout layers are used to prevent overfitting by randomly dropping out (i.e., setting to 0) a fraction of the activations during training. The fraction of activations to drop out is specified by the Dropout rate, which is set to 0.2 in this model.

Softmax Layer:
The final layer is a Dense layer with a Softmax activation function. The purpose of this layer is to produce a probability distribution over the vocabulary for each input sample.#### Results

The results of the text generation using the LSTM and SimpleRNN networks will be shown and compared. This includes the generated text samples, perplexity scores, and visualizations of the network's internal representations.

#### Conclusion

In this project, we demonstrated the use of RNNs for text generation using the Plato corpus. We showed that LSTMs have an advantage over SimpleRNNs in terms of their ability to preserve information over a longer period of time. However, both models were able to generate coherent and semantically meaningful text. This project highlights the versatility and power of RNNs in the field of Natural Language Processing (NLP).


