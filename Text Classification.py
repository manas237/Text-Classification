import tensorflow as tf
from tensorflow import keras
import numpy as np


# Import Data
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)  # Only 10000 most frequent words are kept

review_classification = ["Positive", "Negative"]

# print(len(train_data))
# print(len(test_data))


# Word Dictionary
word_index = data.get_word_index()
word_index = {key: value+3 for key, value in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = {value: key for key, value in word_index.items()}


# Data Preprocessing
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)
# print(len(train_data[0]), len(test_data[15]))


def decode_review(text):
    decoded = " ".join(reverse_word_index.get(i, "?") for i in text)  # separator_string.join(iterable). And dict_name.get("x", ?), works same as dict_name["x"] with additional provision of returning a ? if a key is not found
    return decoded

# print(decode_review(test_data[0]))


# Neural Network Architecture
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))  # Our 88000 most frequent words. Each word is converted to a vector of 16 Dimensions
model.add(keras.layers.GlobalAveragePooling1D())  # Reduces the dimensions of the input data
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))


# Network Optimizer
model.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")


# Network training

# Splitting the training data in 2 parts
x_validate = train_data[:10000]
x_train = train_data[10000:]

y_validate = train_labels[:10000]
y_train = train_labels[10000:]

model.fit(x_train, y_train, batch_size=512, epochs=20,  verbose=1, validation_data=(x_validate, y_validate))  # batch_size is the number of samples to be run through before weights are updated


# Network Accuracy check / Evaluate
loss, acc = model.evaluate(test_data, test_labels)
print(loss, acc)


# Saving the model and Loading the model
model.save("model.h5")

# model = keras.models.load_model("model.h5")


# # Defining Word to Number Encoder for external text file
# def review_encode(s):
#     encoded = [1]
#     for word in s:
#         if word.lower() in word_index:  # .lower() to remove any capital letters and make them lowercase
#             encoded.append(word_index[word.lower()])
#         else:
#             encoded.append(word_index["<UNK>"])
#     return encoded
#
#
# # Opening outside text file
# with open("test.txt", encoding="utf-8") as f:  # using encoding to allow string to be passed in replace()
#     for line in f.readlines():
#         nline = line.replace(",", "").replace(".", "").replace(";", "").replace(":", "").replace("\"", "").replace("(", "").replace(")", "").strip()  # strip() removes the blank spaces at the left and right of the line
#         encode = review_encode(nline)
#         encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
#         prediction = model.predict(encode)
#         print("Review: ", line)
#         print("Encoded line: ", encode)
#         print("Prediction: ", review_classification[int(np.round(prediction))])


# Network Testing / Prediction
test_review = model.predict(test_data[0])
print("Review: ", decode_review(test_data[0]))
print("Actual: ", review_classification[test_labels[0]])
print("Prediction: ", review_classification[int(np.round(test_review[0]))])
