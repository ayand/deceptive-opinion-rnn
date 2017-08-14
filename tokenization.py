import cPickle
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

hotelReviewFile = open("chicago-hotels.json", "r")
reviews = json.load(hotelReviewFile)

totalNumberOfWords = 0

reviewTexts = []

for review in reviews:
    text = review["text"]
    words = text.split(" ")
    totalNumberOfWords += len(words)
    text = text.encode('ascii')
    reviewTexts.append(text)

averageWordNumber = int((totalNumberOfWords / 1600)) + 1
print(str(averageWordNumber))

# Create a tokenizer with the 15000 most common words
reviewTokenizer = Tokenizer(num_words=35000)
reviewTokenizer.fit_on_texts(reviewTexts)
sequences = reviewTokenizer.texts_to_sequences(reviewTexts)

# Make the maximum length of a tokenized sequence equal to the average length of a review
data = pad_sequences(sequences, maxlen=averageWordNumber)

# Save the Tokenizer
with open("review_tokenizer.pickle", "wb") as f:
    cPickle.dump(reviewTokenizer, f)

with open("averageReviewLength.txt", "w") as f:
    f.write(str(averageWordNumber))
    f.close()
