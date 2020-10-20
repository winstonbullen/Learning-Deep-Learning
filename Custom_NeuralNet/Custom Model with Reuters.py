from keras.datasets import reuters
from keras.utils import to_categorical
from keras import models
from keras import layers
import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


input_train = vectorize_sequences(train_data)
input_test = vectorize_sequences(test_data)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

input_validation = input_train[:1000]
split_input_train = input_train[1000:]

target_validation = train_labels[:1000]
split_target_train = train_labels[1000:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(split_input_train,
                    split_target_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(input_validation, target_validation))
test_model = model.evaluate(input_test, test_labels)