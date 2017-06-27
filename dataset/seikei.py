#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


f = open('path.dat', 'r')
data = f.read()
lines = data.split('\n')
hako = []
i = 1000
for line in lines:
    l = line.split(' ')
    l.pop()
    print l
    f_l = map(float,l)
    if len(f_l) == 0:
        da = np.array(hako)
        np.savetxt("{}.csv".format(i),da,delimiter=",")
        i += 1
        hako = []

    else:
        hako.append(f_l)
