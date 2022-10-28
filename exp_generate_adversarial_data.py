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


import pickle 
import argparse
import tensorflow as tf 

from src.utils import DataLoader
from src.models import DenseNet121
from src.adversary import Attacker


parser = argparse.ArgumentParser(
    description = 'What the program does',
    epilog = 'Text at the bottom of help'
) 
parser.add_argument(
    '-s', '--seed', 
    type=int, 
    default=1234, 
    help='Random seed.'
)
parser.add_argument(
    '-o', '--output', 
    type=str, 
    help='Output Path', 
)
parser.add_argument(
    '-a', '--attack', 
    type=str, 
    default='FastGradientMethod', 
    help='Attack [FastGradientMethod]'
)
parser.add_argument(
    '-e', '--epsilon', 
    type=float, 
    default=0.1, 
    help='Attack \epsilon'
)

args = parser.parse_args() 


if __name__ == '__main__': 
    tf.random.set_seed(args.seed)
    dataset = DataLoader(
        image_size=160, 
        batch_size=128, 
        rotation=40, 
        augment=False, 
        store_numpy=True
    )
    network = DenseNet121(
        learning_rate=0.0005, 
        image_size=160, 
        epochs=50
    )
    network.train(dataset)
    
    attack = Attacker(
        attack_type=args.attack, 
        epslison=args.epsilon,
        clip_values=(0,1)
    )
    X = attack.attack(network.network, dataset.X_valid, dataset.y_valid)
    pickle.dump(
        {'X_adv': X, 'args': args}, 
        open(args.output)
    )
    