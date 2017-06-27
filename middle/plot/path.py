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


    df = pd.read_csv('path.csv', sep=",")
    x, y = np.hsplit(df, [1])

    x = np.array(x)
    y = np.array(y)

    #plt.scatter(x[0], y[0], color="red", marker="o", s=100)
    #plt.scatter(x[-1], y[-1], color="red", marker="o", s=100)

    #plt.plot(x, y, color='green', linewidth=1.5)
    #plt.plot(x, y, color='blue', linewidth=1.5)

    plt.plot(x, y, linewidth=1.5, color = 'green')
    plt.plot(x, y, linewidth=1.5, color = 'green')

    plt.legend()
    #plt.xlabel('x', fontsize=18, fontname='serif')
    #plt.ylabel('y', fontsize=18, fontname='serif')
    plt.xlim([-1.0,1.0])
    plt.ylim([-1.0,1.0])
    plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontname='serif')
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontname='serif')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("path15.eps")
    plt.show()


if __name__ == '__main__':
    main()
