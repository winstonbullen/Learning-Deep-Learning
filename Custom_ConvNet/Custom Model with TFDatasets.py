import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

tf.random.set_seed(1234)

BATCH_SIZE = 32
NUM_EPOCHS = 100
IMG_H = IMG_W = 224
IMG_SIZE = 224
LOG_DIR = './CustomModelTFDSLog'
SHUFFLE_BUFFER_SIZE = 1024
IMG_CHANNELS = 3

dataset_name = "cats_vs_dogs"

def preprocess(ds):
    x = tf.image.resize_with_pad(ds['image'], IMG_SIZE, IMG_SIZE)
    x = tf.cast(x, tf.float32)
    x = (x / 127.5) - 1
    return x, ds['label']


def augmentation(image, label):
    image = tf.image.random_brightness(image, .1)
    image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    image = tf.image.random_flip_left_right(image)
    return image, label

def get_dataset(dataset_name):
    train, info_train = tfds.load(dataset_name, split='train[:90%]', with_info=True)
    val, info_val = tfds.load(dataset_name, split='train[:90%]', with_info=True)
    NUM_CLASSES = info_train.features['label'].num_classes
    assert NUM_CLASSES >= info_val.features['label'].num_classes
    NUM_EXAMPLES = info_train.splits['train'].num_examples * 0.9
    IMG_H, IMG_W, IMG_CHANNELS = info_train.features['image'].shape
    train = train.map(preprocess).repeat().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    train = train.map(augmentation)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)
    val = val.map(preprocess).repeat().batch(BATCH_SIZE)
    val = val.prefetch(tf.data.experimental.AUTOTUNE)
    return train, info_train, val, info_val, IMG_H, IMG_W, IMG_CHANNELS, NUM_CLASSES, NUM_EXAMPLES

train, info_train, val, info_val, IMG_H, IMG_W, IMG_CHANNELS, NUM_CLASSES, NUM_EXAMPLES = \
    get_dataset(dataset_name)

tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_DIR,
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      write_grads=True,
                                                      batch_size=BATCH_SIZE,
                                                      write_images=True)

def create_model():
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),
                                 tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                                 tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                 tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                                 tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                 tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                                 tf.keras.layers.Dropout(rate=0.3),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dropout(rate=0.3),
                                 tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')])
    return model

def scratch(train, val, learning_rate):
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5)
    model.fit(train,
              epochs=NUM_EPOCHS,
              steps_per_epoch=int(NUM_EXAMPLES/BATCH_SIZE),
              validation_data=val,
              validation_steps=1,
              validation_freq=1,
              callbacks=[tensorboard_callback, earlystop_callback])
    return model

# Run tensorboard --logdir ./CustomModelLog in Terminal

learning_rate = 0.001

model = scratch(train, val, learning_rate)