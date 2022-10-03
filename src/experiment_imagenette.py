# MIT License
# 
# Copyright (c) 2022 Gregory Ditzler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense 
import tensorflow_datasets as tfds


# globals -- or convert to args 
NUM_CLASSES = 10
IMAGE_SIZE = 64
BATCH_SIZE = 256 
EPOCHS = 100
LEARNING_RATE = 0.001
KERNEL_SIZE = (3, 3)
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)


def load_data(): 
    """
    """
    data, _ = tfds.load("imagenette/160px-v2", with_info=True, as_supervised=True)
    train_data, valid_data = data['train'], data['validation']
    train_dataset = train_data.map(
        lambda image, label: (tf.image.resize(image, (IMAGE_SIZE,IMAGE_SIZE )), label))

    validation_dataset = valid_data.map(
        lambda image, label: (tf.image.resize(image, (IMAGE_SIZE,IMAGE_SIZE )), label)
    )


    data_augmentation = tf.keras.models.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
    ])

    # Create a tf.data pipeline of augmented images (and their labels)
    # TO-DO: Compress this into fewer lines of code  
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
    train_dataset_2 = train_dataset.map(lambda x, y: (data_augmentation(x), y))
    train_dataset_3 = train_dataset.map(lambda x, y: (data_augmentation(x), y))

    X_train, y_train = list(map(lambda x: x[0], train_dataset)), list(map(lambda x: x[1], train_dataset))
    X_train_2, y_train_2 = list(map(lambda x: x[0], train_dataset)), list(map(lambda x: x[1], train_dataset_2))
    X_train_3, y_train_3 = list(map(lambda x: x[0], train_dataset)), list(map(lambda x: x[1], train_dataset_3))
    X_valid, y_valid = list(map(lambda x: x[0], validation_dataset)), list(map(lambda x: x[1], validation_dataset))
    
    X_train = np.concatenate((np.array(X_train), np.array(X_train_2), np.array(X_train_3)))
    y_train = np.concatenate((np.array(y_train), np.array(y_train_2), np.array(y_train_3)))
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=40, height_shift_range=0.2)
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_ds = tf.keras.preprocessing.image.NumpyArrayIterator(
        x=np.array(X_train), 
        y=np.array(y_train), 
        image_data_generator=train_datagen,
        batch_size=BATCH_SIZE
    )

    valid_ds = tf.keras.preprocessing.image.NumpyArrayIterator(
        x=np.array(X_valid), 
        y=np.array(y_valid), 
        image_data_generator=valid_datagen,
        batch_size=BATCH_SIZE
    )
    return train_ds, valid_ds  



train_ds, valid_ds = load_data()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same', input_shape=INPUT_SHAPE), 
    tf.keras.layers.Conv2D(32, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same'), 
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Conv2D(64, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same'), 
    tf.keras.layers.Conv2D(64, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same', strides=(2, 2)), 
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Conv2D(128, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same'), 
    tf.keras.layers.Conv2D(128, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same', strides=(2, 2)), 
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(256, activation='elu', kernel_initializer='he_uniform'), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(128, activation='elu', kernel_initializer='he_uniform'), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(10, activation='softmax')
])


# set up the optimizer 
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE, 
    beta_1=0.9, 
    beta_2=0.999
)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy()
]
model.compile(
    optimizer=optimizer, 
    loss=loss, 
    metrics=metrics,
)

# train / evaluate the model. 
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=250,
    # callbacks=[scheduler],
    verbose=1
)
model.save('MultiScaleAdversarialAttacks/imagenette_cnn_2.h5')
np.save('MultiScaleAdversarialAttacks/imagenette_cnn_history_2.npy', history.history)

