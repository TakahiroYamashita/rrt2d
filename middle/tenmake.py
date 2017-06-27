# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pandas as pd

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from matplotlib import pyplot as plt
#from train import RRT
import train_1 as R
import csv
import os


def main():

    fname = "plot/ten1/ten1.csv"
    os.remove(fname)
    f = open(fname, 'ab')
    csvWriter = csv.writer(f)
    csvWriter.writerow('')

    N = 900
    N_test = 100
    n_in = 4
    n_units=30
    n_out = 2

    sumtime = 0

    ## load_data ##
    df = pd.read_csv('data_1.csv', sep=",")
    rrt_data, rrt_label = np.hsplit(df, [4])
    x_train, x_test = np.split(rrt_data, [N])
    y_train, y_test = np.split(rrt_label, [N])
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    ## model ##
    rrt = R.RRT(n_in, n_units, n_out)
    model = L.Classifier(rrt, lossfun=F.mean_squared_error)


    for i in range(N_test):
        x_test_train = np.array([x_test[i]], dtype=np.float32)
        #y_test_train = np.array([y_test[i]], dtype=np.float32)

        #start, goal = np.hsplit(x_test_train, [2])
        #t = Variable(y_test_train)
        serializers.load_npz('middle_1.model', model)
        model.compute_accuracy = False
        x = Variable(x_test_train)
        m = rrt(x)


        ## tenの書き込み ##

        #dataset_s = np.hstack((start[0][0], start[0][1]))
        dataset_m = np.hstack((m[0][0].data, m[0][1].data))
        #dataset_g = np.hstack((goal[0][0], goal[0][1]))

        #csvWriter.writerow(dataset_s)
        csvWriter.writerow(dataset_m)
        #csvWriter.writerow(dataset_g)

    f.close()


    ## plot ##

    ## 障害物 ##
    plt.gca().add_patch(plt.Rectangle(xy=[-0.5, 0.2], width=1.3, height=0.5, edgecolor='blue', facecolor='deepskyblue'))
    plt.gca().add_patch(plt.Rectangle(xy=[-0.2, -0.5], width=0.9, height=0.4, edgecolor='blue', facecolor='deepskyblue'))
    plt.gca().add_patch(plt.Rectangle(xy=[-0.8, -0.8], width=0.3, height=0.6, edgecolor='blue', facecolor='deepskyblue'))

    df = pd.read_csv('plot/ten1/ten1.csv', sep=",")
    x, y = np.hsplit(df, [1])

    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.scatter(x, y, color='navy', linewidth=2.0)

    #plt.xlabel("x", fontsize=22, fontname='serif')
    #plt.ylabel("y", fontsize=22, fontname='serif')

    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])
    plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontname='serif')
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontname='serif')
    plt.tick_params(labelsize=18)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("plot/ten1/ten1.eps")

    plt.show()


if __name__ == '__main__':
    main()
