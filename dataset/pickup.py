# -*- coding: utf-8 -*-

from __future__ import print_function
import pandas as pd
import numpy as np
import csv


def main():

	N = 11  #データ数
	num = 1

	for i in range(N):
		fname = str(i)+".csv"
		df = pd.read_csv(fname, sep=",", names=['a','b'])
		data = np.array(df)
		#print("data", data)

		d_sum = 0

		a = data[0]-data[-1]
		#print("a", a)
		p = np.sqrt(np.power(a[0], 2)+np.power(a[1], 2))
		#print("p", p)

		lines = data.shape[0]
		#print("lines", lines)

		#if (lines != 25):
			#print("not")



		for line in range(lines-1):
			a_ = data[line]-data[line+1]
			#print("a_", a_)
			d = np.sqrt(np.power(a_[0], 2) + np.power(a_[1], 2))
			#print("d", d)
			d_sum += d

		#print("d_sum", d_sum)
		#print("p", np.sqrt(2)*p)
		#print("")


		if (d_sum > np.sqrt(2)*p):
			print(num, "", i, "not")
			num+=1
		"""
		else:
			data_x = np.hstack((data[0], data[-1]))

			if (lines%2==0):
				middle_x = (data[0][0]+data[-1][0])/2
				middle_y = (data[0][1]+data[-1][1])/2
				data_y = np.hstack((middle_x, middle_y))

			else:
				middle = (lines-1)/2
				data_x = np.hstack((data[0], data[-1]))
				data_y = data[middle]

			dataset_n = np.hstack((data_x, data_y))
			csvWriter.writerow(dataset_n)
			"""

	#f.close()



if __name__ == '__main__':
	main()
