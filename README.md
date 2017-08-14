# Keras RNN for Discriminating Between Truthful and Deceptive Hotel Reviews

Source of Data (converted into JSON format):
https://www.kaggle.com/rtatman/deceptive-opinion-spam-corpus

The purpose of this project is to use a recurrent neural network constructed by the Keras deep learning framework to determine whether or not a review for a restaurant is spam. The text of the reviews is first tokenized based on the most frequent occurrences of certain words. The generation of such a tokenizer is done in tokenization.py. The tokenizer generated here is used in the rnnTraining.py file to train a recurrent neural network using an LSTM unit in order to discriminate between truthful and deceptive reviews. Parameters tweaked include:
* the size of the considered vocabulary
* size of the vectors generated for each word
* Dropout rate
* Number of units in the hidden layers of the neural network as well as number of hidden layers (usually varies between 1 and 3)
* Optimization method (chose between adam and RMS propagation methods)
* Batch size
* Number of training epochs

The highest validation accuracy yielded has been 69.25%.
