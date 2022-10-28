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

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod

class Attacker: 
    def __init__(self, attack_type:str='FastGradientMethod', epsilon:float=0.1, clip_values:tuple=(0, 1)): 
        self.epsilon = epsilon
        self.attack_type = attack_type
        self.clip_values = clip_values
        
    def attack(self, network, X, y=None): 
        classifier = KerasClassifier(model=network, clip_values=self.clip_values)
        
        if self.attack_type == 'FastGradientMethod': 
            adv_crafter = FastGradientMethod(classifier, eps=self.epsilon)
            Xadv = adv_crafter.generate(x=X)
        else: 
            ValueError('Unknown attack type')
        
        return Xadv