# -*- coding: utf-8 -*-

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    ## 障害物 ##
    plt.gca().add_patch(plt.Rectangle(xy=[-0.5, 0.2], width=1.3, height=0.5, edgecolor='blue', facecolor='deepskyblue'))
    plt.gca().add_patch(plt.Rectangle(xy=[-0.2, -0.5], width=0.9, height=0.4, edgecolor='blue', facecolor='deepskyblue'))
    plt.gca().add_patch(plt.Rectangle(xy=[-0.8, -0.8], width=0.3, height=0.6, edgecolor='blue', facecolor='deepskyblue'))


    #df = pd.read_csv('ten1/ten1.csv', sep=",")
    #x, y = np.hsplit(df, [1])

    #plt.rcParams['font.family'] = 'Times New Roman'
    #plt.scatter(x, y, color='navy', linewidth=2.0)

    #plt.xlabel("x axis [m]", fontname='serif', fontsize=17)
    #plt.ylabel("y axis [m]", fontname='serif', fontsize=17)

    plt.xlim([-1.0,1.0])
    plt.ylim([-1.0,1.0])
    plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontname='serif')
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontname='serif')
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.savefig("ten1/ten.eps")
    plt.tick_params(labelsize=18)
    plt.savefig("environment.eps")

    plt.show()


if __name__ == '__main__':
    main()
