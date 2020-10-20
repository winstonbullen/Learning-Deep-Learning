from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


input_train = vectorize_sequences(train_data)
input_test = vectorize_sequences(test_data)

target_train = np.asarray(train_labels).astype('float32')
target_test = np.asarray(test_labels).astype('float32')

input_validation = input_train[:10000]
split_input_train = input_train[10000:]

target_validation = target_train[:10000]
split_target_train = target_train[10000:]

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(split_input_train,
                    split_target_train,
                    epochs=5,
                    batch_size=512,
                    validation_data=(input_validation, target_validation))