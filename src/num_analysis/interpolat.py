"""Iterpolations
* :function:`.chebyshev_pts`
* :function:`.lagrange`
* :function:`.lebesgue`
* :function:`.`
"""
#%%

# Plot
import matplotlib.pyplot as plt

# General
import math
import numpy as np


def chebyshev_pts(n):
	pi = math.pi
	tt = np.linspace(0,pi,n)
	zz = np.exp(complex(0, 1)*tt)
	x = [ele.real for ele in zz]
	return np.array(x)


def lagrange(xinter, yinter, fnc=None):
	"""
    Summary:    From the intersection coordinates (xinter, yinter), thins function generates
    --------    and plots the lagrange interpolation curve based on the .

    Input:      - xinter: x intercept (i.e. chebyshev or equi-dist)
    ------       - yinter: xinter correspondent points in domain y
				- func: case the funciont is known

    Output:     - The Lagrange function plot for the [a,b] interval in blue and
	-------     fnc in red (case it is known).


	Example1:	if f(x) is known
	--------	xinter :  np.linspace(a, b, 1,n+1)	(numpy.arrays)
				yinter = f(xi)
				p = lagrange(10, x, xinter, yinter, func = None )
	"""

	a, b = xinter.min(), xinter.max()
	x = np.linspace(a, b, int(b-a)*500) #(continuous space domain in x)
	p = np.zeros(int(b - a)*500)

	for k in range(len(xinter)):
		L = np.ones(int(b-a)*500)
		for i in range(len(xinter)):
			if i!=k:
				L = L*(x-xinter[i])/(xinter[k]-xinter[i])
		p = p+L*yinter[k]

	n = len(xinter)
	label = "Lagrange interpolation for n= %d."%n

	plt.plot(x,p, 'b', label= 'interpolat')
	if fnc:
		y = fnc(x)
		plt.plot(x, y, 'r', label= 'fnc')
	plt.plot(xinter, yinter, 'ok')
	plt.xlabel( 'x')
	plt.ylabel('y')
	plt.title(label)
	plt.legend()
	plt.grid()


	plt.show()
	return



def lebesgue(xinter):
	"""
    Summary:    It plots the Lebesgue funtion for a given interpolation points to visualize the
	--------    the Lesbegue constant growth.

    Input:      - xinter: x intercept (i.e. chebyshev or equi-dist)
    ------

    Output:     - The Lebesgue funtion plot for the [-1,1] interval.


	Example1:	if f(x) is known
	--------	xinter :  interpolation points

	"""
	a, b = xinter.min(), xinter.max()
	x = np.linspace(a, b, int(b-a)*500) #(continuous space domain in x)
	p = np.zeros(int(b - a)*500)
	for k in range(len(xinter)):
		L = np.ones(int(b-a)*500)
		for i in range(len(xinter)):
			if i!=k:
				L = L*(x-xinter[i])/(xinter[k]-xinter[i])
		p = p + np.abs(L)

	n = len(xinter)
	label = "Lebesgue function n= %d interpolation points."%n

	plt.plot(x,p, 'b')
	plt.xlabel( 'x')
	plt.ylabel('y')
	plt.title(label)
	plt.legend()
	plt.grid()
	plt.show()
	return


def f(x):
    f_ = 1/(1+25*x**2)
    return f_

n=10

xi = np.linspace(-.5,.5,n)
yi = f(xi)

x_ch = chebyshev_pts(n)
y_ch = f(x_ch)

plag = lagrange(xi, yi, f)
ples = lebesgue(xi)
plag = lagrange(x_ch, y_ch, f)
ples = lebesgue(x_ch)







# %%
