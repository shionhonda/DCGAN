#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L


def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h

class Generator(chainer.Chain):

    def __init__(self, n_hidden, bottom_height=12, bottom_width=9, ch=512, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_height = bottom_height
        self.bottom_width = bottom_width

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_height * bottom_width * ch,
                               initialW=w)
            self.c1_0 = L.Convolution2D(ch, ch*2, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch//2, ch*2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch//2, ch, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(ch//4, ch, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch//4, ch//2, 3, 1, 1, initialW=w)
            self.c3_1 = L.Convolution2D(ch//8, 32, 4, 2, 1, initialW=w)
            self.c4_0 = L.Convolution2D(8, 3, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(bottom_height * bottom_width * ch)
            self.bn1_0 = L.BatchNormalization(ch*2)
            self.bn1_1 = L.BatchNormalization(ch*2)
            self.bn2_0 = L.BatchNormalization(ch)
            self.bn2_1 = L.BatchNormalization(ch)
            self.bn3_0 = L.BatchNormalization(ch//2)
            self.bn3_1 = L.BatchNormalization(32)

    def make_hidden(self, batchsize):
        hidden = numpy.random.normal(0, 0.5, (batchsize, self.n_hidden, 1, 1))
        return hidden.astype(numpy.float32)

    def __call__(self, z):
        h = F.reshape(F.leaky_relu(self.bn0(self.l0(z))),
                      (len(z), self.ch, self.bottom_height, self.bottom_width))
        h = F.dropout(h, 0.5)
        h = F.leaky_relu(F.depth2space(self.bn1_0(self.c1_0(h)), 2))
        h = F.dropout(h, 0.4)
        h = F.leaky_relu(F.depth2space(self.bn1_1(self.c1_1(h)), 2))
        h = F.dropout(h, 0.2)
        h = F.leaky_relu(F.depth2space(self.bn2_0(self.c2_0(h)), 2))
        h = F.leaky_relu(F.depth2space(self.bn2_1(self.c2_1(h)), 2))
        h = F.leaky_relu(F.depth2space(self.bn3_0(self.c3_0(h)), 2))
        h = F.leaky_relu(F.depth2space(self.bn3_1(self.c3_1(h)), 2))
        x = F.sigmoid(self.c4_0(h))
        return x



class Discriminator(chainer.Chain):

    def __init__(self, bottom_height=12, bottom_width=9, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_height * bottom_width * ch, 1, initialW=w)
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        ratio = 0.5
        h = add_noise(x)
        h = F.leaky_relu(add_noise(self.c0_0(h)))
        h = F.dropout(h, ratio)
        h = F.leaky_relu(add_noise(self.bn0_1(self.c0_1(h))))
        h = F.dropout(h, ratio)
        h = F.leaky_relu(add_noise(self.bn1_0(self.c1_0(h))))
        h = F.dropout(h, ratio)
        h = F.leaky_relu(add_noise(self.bn1_1(self.c1_1(h))))
        h = F.dropout(h, ratio)
        h = F.leaky_relu(add_noise(self.bn2_0(self.c2_0(h))))
        h = F.dropout(h, ratio)
        h = F.leaky_relu(add_noise(self.bn2_1(self.c2_1(h))))
        h = F.dropout(h, ratio)
        h = F.leaky_relu(add_noise(self.bn3_0(self.c3_0(h))))
        h = F.dropout(h, ratio)
        return F.dropout(self.l4(h), ratio)
