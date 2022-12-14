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

import tensorflow as tf 
from .utils import DataGenFusion

class VanillaCNN:
    
    def __init__(self, learning_rate:float=0.0005, image_size:int=160, epochs:int=50):
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.histories = []
        self.epochs = epochs
        KERNEL_SIZE = (3, 3)
        
        input = tf.keras.layers.Input(shape=(image_size, image_size, 3), name="image")
        x = tf.keras.layers.Conv2D(32, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same')(input)
        x = tf.keras.layers.Conv2D(32, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same')(x) 
        x = tf.keras.layers.MaxPooling2D((2, 2))(x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv2D(64, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same')(x) 
        x = tf.keras.layers.Conv2D(64, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same', strides=(2, 2))(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv2D(128, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same')(x) 
        x = tf.keras.layers.Conv2D(128, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same', strides=(2, 2))(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x)

        model_res = tf.keras.Model(inputs=input, outputs=predictions)

        model_res.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  metrics=['accuracy'])
        self.network = model_res
    
    def train(self, dataset): 
        history = self.network.fit(
            dataset.train_ds,
            validation_data=dataset.valid_ds,
            epochs=self.epochs,
            verbose=1
        )
        self.histories.append(history) 
 

class DenseNet121:
    
    def __init__(self, learning_rate:float=0.0005, image_size:int=160, epochs:int=50):
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.histories = []
        self.epochs = epochs
        
        model_res = tf.keras.applications.densenet.DenseNet121(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_size, self.image_size, 3)
        )
        for layer in model_res.layers:
            layer.trainable = True

        x = tf.keras.layers.Flatten()(model_res.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x)

        model_res = tf.keras.Model(inputs=model_res.input, outputs=predictions)

        model_res.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  metrics=['accuracy'])
        self.network = model_res
    
    def train(self, dataset): 
        history = self.network.fit(
            dataset.train_ds,
            validation_data=dataset.valid_ds,
            epochs=self.epochs,
            verbose=1
        )
        self.histories.append(history) 
        

class MultiResolutionNetwork: 
    def __init__(self, image_sizes:list=[32,64,160], learning_rate:float=0.0005, epochs:int=10): 
        
        # require that we only have three image sizes... no more... no less. 
        if len(image_sizes) != 3: 
            raise(ValueError(
                ''.join(['image_sizes must be of length 3. Currently, len(', 
                str(image_sizes), ') = ', str(len(image_sizes))])
            ))
        
        
        self.image_sizes = image_sizes
        self.learning_rate = learning_rate
        self.histories = []
        self.epochs = epochs
        
        # need to set up an iterator than can be used to rename a layer of the network
        # because layer names need to be changed since we are going to use the same type 
        # of backbone. 
        k = 0 
        
        # MODEL 01 - image_size[0]
        model_01 = tf.keras.applications.densenet.DenseNet121(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_sizes[0], self.image_sizes[0], 3)
        )
        for layer in model_01.layers:
            layer.trainable = True
            layer._name = layer._name+'_'+str(k)
            k += 1

        # MODEL 02 - image_size[1]
        model_02 = tf.keras.applications.densenet.DenseNet121(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_sizes[1], self.image_sizes[1], 3)
        )
        for layer in model_02.layers:
            layer.trainable = True
            layer._name = layer._name+'_'+str(k)
            k += 1

        # MODEL 03 - image_size[2]
        model_03 = tf.keras.applications.densenet.DenseNet121(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_sizes[2], self.image_sizes[2], 3)
        )
        for layer in model_03.layers:
            layer.trainable = True
            layer._name = layer._name+'_'+str(k)
            k += 1
            
        
        x = tf.keras.layers.Flatten()(model_01.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        y = tf.keras.layers.Flatten()(model_02.output)
        y = tf.keras.layers.Dense(1024, activation='relu')(y)
        z = tf.keras.layers.Flatten()(model_03.output)
        z = tf.keras.layers.Dense(1024, activation='relu')(z)

        x = tf.keras.layers.concatenate([x, y, z], name="concat_layer")
        
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x)
        
        self.network = tf.keras.models.Model(
            inputs=[model_01.input, model_02.input, model_03.input], 
            outputs=predictions
        ) 
        self.network.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
            metrics=['accuracy']
        )
        
    def train(self, dataset): 
        history = self.network.fit(
            dataset.train_ds,
            validation_data=dataset.valid_ds,
            epochs=self.epochs,
            verbose=1
        )
        self.histories.append(history)  


          