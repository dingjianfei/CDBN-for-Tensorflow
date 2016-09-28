# Copyright 2016 Sanghoon Yoon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convolutional Deep Belief Network for Tensorflow"""
from core.model import Model
from crbm import CRBM


class CDBN(Model):
    def __init__(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.crbms = [
            CRBM('layer1', 300, 3, 10, 32, 3, self.batch_size, self.learning_rate, True),
            CRBM('layer2', 100, 32, 10, 64, 2, self.batch_size, self.learning_rate, False)
        ]
        self._layer_iterator = self.__ops_generator()

    @property
    def input(self):
        return self.crbms[0].input

    @property
    def output(self):
        return self.crbms[-1].output

    @property
    def training_finished(self):
        return False if self.training_layer is not None else True

    @property
    def ops(self):
        return self.training_layer.ops

    def build_graphs(self):
        for idx, crbm in enumerate(self.crbms):
            if idx != 0:
                crbm.set_input(self.crbms[idx - 1].output)
            crbm.build_graphs()

    @property
    def training_layer(self):
        try:
            return self._layer_iterator.next()
        except StopIteration:
            return None

    def __ops_generator(self):
        for crbm in self.crbms:
            while not crbm.training_finished:
                yield crbm
