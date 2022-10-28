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
import tensorflow_datasets as tfds

class DataLoader(): 
    """_summary_
    """
    def __init__(self, image_size:int=160, batch_size:int=128, rotation:int=40, augment:bool=False, store_numpy:bool=False): 
        self.n_classes = 10 
        self.image_size = image_size
        self.batch_size = batch_size
        self.input_shape = (self.image_size, self.image_size, 3)
        self.rotation = rotation
        self.augment = augment
        self.train_ds, self.valid_ds = None, None  
        self.y_valid, self.y_train = None, None 
        self.X_valid, self.X_train = None, None 
        self.store_numpy = store_numpy
        self._load_data()
    
    
    def _load_data(self): 
        data, _ = tfds.load("imagenette/160px-v2", with_info=True, as_supervised=True)
        train_data, valid_data = data['train'], data['validation']
        train_dataset = train_data.map(
            lambda image, label: (tf.image.resize(image, (self.image_size, self.image_size)), label)
        )
        validation_dataset = valid_data.map(
            lambda image, label: (tf.image.resize(image, (self.image_size, self.image_size)), label)
        )
        
        if self.augment: 
            data_augmentation = tf.keras.models.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ])
            train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))


        X_train, y_train = list(map(lambda x: x[0], train_dataset)), \
            list(map(lambda x: x[1], train_dataset))
        X_valid, y_valid = list(map(lambda x: x[0], validation_dataset)), \
            list(map(lambda x: x[1], validation_dataset))

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, rotation_range=self.rotation, height_shift_range=0.2
        )
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        self.train_ds = tf.keras.preprocessing.image.NumpyArrayIterator(
            x=np.array(X_train), 
            y=np.array(y_train), 
            image_data_generator=train_datagen,
            batch_size=self.batch_size
        )

        self.valid_ds = tf.keras.preprocessing.image.NumpyArrayIterator(
            x=np.array(X_valid), 
            y=np.array(y_valid), 
            image_data_generator=valid_datagen,
            batch_size=self.batch_size
        )
        if self.store_numpy: 
            self.X_train = np.array(X_train)/255.
            self.y_train = np.array(y_train)
            self.X_valid = np.array(X_valid)/255. 
            self.y_valid = np.array(y_valid)


