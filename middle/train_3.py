# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import numpy as np
import pandas as pd

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from matplotlib import pyplot as plt
import time


class RRT(Chain):
    def __init__(self, n_in, n_units, n_out, train=True):
        super(RRT, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_units),
            l4=L.Linear(n_units, n_units),
            l5=L.Linear(n_units, n_units),
            l6=L.Linear(n_units, n_out)
        )
        self.train=train

    def __call__(self, x):
        h0=F.relu(self.l1(x))
        h1=F.relu(self.l2(h0))
        h2=F.relu(self.l3(h1))
        h3=F.relu(self.l4(h2))
        h4=F.relu(self.l5(h3))
        y=self.l6(h4)
        return y


def main():

    n_in = 4
    n_units =30
    n_out = 2

    n_epoch = 20
    batchsize = 50
    N = 900
    N_test = 100

    ## load_data ##
    df = pd.read_csv('data_3.csv', sep=",")
    rrt_data, rrt_label = np.hsplit(df, [4])
    x_train, x_test = np.split(rrt_data, [N])
    y_train, y_test = np.split(rrt_label, [N])

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)


    ## model ##
    rrt = RRT(n_in, n_units, n_out)
    model = L.Classifier(rrt, lossfun=F.mean_squared_error)
    model.compute_accuracy = False
    optimizer = optimizers.Adam()
    optimizer.setup(model)


    ## Learn ##
    train_losses = []
    test_losses = []

    for epoch in range(n_epoch):
        perm = np.random.permutation(N)
        sum_loss = 0
        sum_loss_test = 0

        ## train ##
        for i in range(0, N, batchsize):
            x = Variable(x_train[perm[i:i + batchsize]])
            t = Variable(y_train[perm[i:i + batchsize]])
            #y = rrt(x)
            optimizer.update(model, x, t)
            loss = model.loss
            #print("loss", loss.data)
            sum_loss += loss.data * batchsize

        average_loss = sum_loss / N
        train_losses.append(average_loss)


        ## test ##
        for i in range(N_test):
            x_test_train = np.array([x_test[i]], dtype=np.float32)
            y_test_train = np.array([y_test[i]], dtype=np.float32)
            x_ = Variable(x_test_train)
            t_ = Variable(y_test_train)
            #y = rrt(x_)
            loss = model(x_, t_)
            #print("loss", loss.data)
            sum_loss_test += loss.data

        ave_test_loss = sum_loss_test / N_test
        test_losses.append(ave_test_loss)

    serializers.save_npz('middle_3.model', model)
    # serializers.save_npz('test.state', optimizer)


    """
    ## 表示 ##
    print("train_losses")
    for a in range(epoch + 1):
        print(train_losses[a])
    print("")
    print("test_losses")
    for a in range(epoch+1):
        print(test_losses[a])
    """

    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.legend(fontsize=20)
    plt.xlim([0, 19])
    plt.ylim([0, 0.04])
    plt.xlabel("epoch count", fontsize=20, fontname='serif')
    plt.ylabel("loss", fontsize=20, fontname='serif')
    plt.tick_params(labelsize=18)
    plt.xticks([0, 4, 8, 12, 16], fontname='serif')
    plt.yticks([0.02, 0.04, 0.06, 0.08, 0.10], fontname='serif')
    plt.savefig("fig3/loss3.eps")
    plt.savefig("fig3/loss3.png")
    plt.show()



if __name__ == '__main__':
    #for f in range(10):
    #start = time.time()
    main()
    #elapsed_time = time.time() - start
    #print("elapsed_time:{0}".format(elapsed_time))
