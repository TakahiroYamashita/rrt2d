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
import time
import csv
import os
import train_1 as m1
import train_2 as m2
import train_3 as m3



def main():

    #fname = "plot/path.csv"
    #os.remove(fname)
    #f = open(fname, 'ab')
    #csvWriter = csv.writer(f)

    N = 900
    N_test = 100
    n_in=4
    n_units=30
    n_out = 2

    sumtime = 0

    ## load_data ##
    df = pd.read_csv('m.csv', sep=",")
    rrt_data, rrt_label = np.split(df, [4], axis=1)
    x_train, x_test = np.split(rrt_data, [N])
    y_train, y_test = np.split(rrt_label, [N])
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    ## model ##
    rrt1 = m1.RRT(n_in, n_units, n_out)
    model1 = L.Classifier(rrt1, lossfun=F.mean_squared_error)
    serializers.load_npz('middle_1.model', model1)
    model1.compute_accuracy = False

    rrt2 = m2.RRT(n_in, n_units, n_out)
    model2 = L.Classifier(rrt2, lossfun=F.mean_squared_error)
    serializers.load_npz('middle_2.model', model2)
    model2.compute_accuracy = False

    rrt3 = m3.RRT(n_in, n_units, n_out)
    model3 = L.Classifier(rrt3, lossfun=F.mean_squared_error)
    serializers.load_npz('middle_3.model', model3)
    model3.compute_accuracy = False


    for i in range(N_test):
        x_test_train = np.array([x_test[i]], dtype=np.float32)
        #y_test_train = np.array([y_test[i]], dtype=np.float32)
        #print("x_test", x_test_train)
        #print("m_test", y_test_train)
        #print("")

        start, goal = np.hsplit(x_test_train, [2])

        x_1 = Variable(x_test_train)
        #t = Variable(y_test_train)


        starts = time.time()

        m_1 = rrt1(x_1)
        #print("m_1", m_1.data)
        x_2 = np.hstack((start, m_1.data))
        #print("x_2", x_2)
        x_2 = Variable(x_2)
        m_2 = rrt2(x_2)
        #print("m_2", m_2.data)
        x_3 = np.hstack((m_1.data, goal))
        # print("x_3", x_3)
        x_3 = Variable(x_3)
        m_3 = rrt3(x_3)
        # print("m_3", m_3.data)

        elapsed_time = time.time() - starts
        #print("elapsed_time:{0}".format(elapsed_time))

        sumtime += elapsed_time

        #print(start[0][0], "", end="")
        #print(start[0][1])

        #print(m_2[0][0].data, "", end="")
        #print(m_2[0][1].data)

        #print(m_1[0][0].data, "", end="")
        #print(m_1[0][1].data)

        #print(m_3[0][0].data, "", end="")
        #print(m_3[0][1].data)

        #print(goal[0][0], "", end="")
        #print(goal[0][1])

        #print("")

    print("sumtime", sumtime)


        #csvWriter.writerow('')
        #dataset_s = np.hstack((start[0][0], start[0][1]))
        #dataset_m2 = np.hstack((m_2[0][0].data, m_2[0][1].data))
        #dataset_m1 = np.hstack((m_1[0][0].data, m_1[0][1].data))
        #dataset_m3 = np.hstack((m_3[0][0].data, m_3[0][1].data))
        #dataset_g = np.hstack((goal[0][0], goal[0][1]))

        #csvWriter.writerow(dataset_s)
        #csvWriter.writerow(dataset_m2)
        #csvWriter.writerow(dataset_m1)
        #csvWriter.writerow(dataset_m3)
        #csvWriter.writerow(dataset_g)


    #f.close()


if __name__ == '__main__':
    main()

