"""Iterpolations
* :class:`.BarycentricRational`
* :function:`.chebyshev_pts`
* :function:`.lagrange`
* :function:`.lebesgue`
"""
#%%

# Plot
import matplotlib.pyplot as plt

# General
import math
import numpy as np


class BarycentricRational:
    """A class representing a rational function in barycentric representation.
    Args:
        z (array): the interpolation nodes
        f (array): the values at the interpolation nodes
        w (array): the weights
    The rational function has the interpolation property r(z_j) = f_j at all
    nodes where w_j != 0.
    """
    def __init__(self, z, f, w):
        if not (len(z) == len(f) == len(w)):
            raise ValueError('arrays z, f, and w must have the same length')
        self.nodes = np.asanyarray(z)
        self.values = np.asanyarray(f)
        self.weights = np.asanyarray(w)

    def __call__(self, x):
        """Evaluate rational function at all points of `x`."""
        zj,fj,wj = self.nodes, self.values, self.weights

        xv = np.asanyarray(x).ravel()
        if len(xv) == 0:
            return np.empty(np.shape(x), dtype=xv.dtype)
        D = xv[:,None] - zj[None,:]
        # find indices where x is exactly on a node
        (node_xi, node_zi) = np.nonzero(D == 0)

        one = xv[0] * 0 + 1     # for proper dtype when using mpmath

        with np.errstate(divide='ignore', invalid='ignore'):
            if len(node_xi) == 0:       # no zero divisors
                C = np.divide(one, D)
                r = C.dot(wj * fj) / C.dot(wj)
            else:
                # set divisor to 1 to avoid division by zero
                D[node_xi, node_zi] = one
                C = np.divide(one, D)
                r = C.dot(wj * fj) / C.dot(wj)
                # fix evaluation at nodes to corresponding fj
                # TODO: this is only correct if wj != 0
                r[node_xi] = fj[node_zi]

        if np.isscalar(x):
            return r[0]
        else:
            r.shape = np.shape(x)
            return r



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
				L = L*(x-xinter[i])/(xinter[k]-xinter[i])  # the points x throughx'
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

	# adjust return for nodes, values and weights
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

	# adjust return for nodes, values and weights
	return

def floater_hormann(d, xinter, yinter, fnc=None):
    """
    Summary:    From the intersection coordinates (xinter, yinter), thins function generates
    --------    and plots using Floater-Hormann rational interpolant curve based on the .

    Input:      - xinter: x intercept (i.e. chebyshev or equi-dist)
    ------      - yinter: xinter correspondent points in domain y
                - func: case the funciont is known
                - d: the window size

    Output:     - The Lagrange function plot for the [a,b] interval in blue and
    -------     fnc in red (case it is known).


    Example1:   if f(x) is known
    --------    xinter :  np.linspace(a, b, 1,n+1) (numpy.arrays)
                yinter = f(xi)
                p = lagrange(10, x, xinter, yinter, func = None )
    """
    # Nodes limits
    a, b = xinter.min(), xinter.max()
    n = len(xinter)

    # weights
    weights = np.zeros(n + 1)

    for i in range(len(xinter)):
        Ji = range(max(0, i-d), min(i, n-1-d) + 1)
        w = 0.0
        for k in Ji:
            w += np.prod([1.0 / abs(xinter[i] - xinter[j]) for j in range(k, k+d+1) if j != i])
        weights[i] = (-1.0)**(i-d) * w

    # Interpolation
    x_ = np.linspace(a, b, int(b-a)*50*n) # (continuous space domain in x)
    x = np.array([i for i in x_ if i not in xinter])

    num = np.ones(len(x))
    den = np.ones(len(x))

    for i in range(len(xinter)):

        num += weights[i]*yinter[i]/(x-xinter[i])
        den += weights[i]/(x-xinter[i])

    p = num/den


    label = "Floater-Hormann interpolation for n= {} and d ={}".format(n, d)

    plt.plot(x,p, 'b', label= 'interpolat')
    if fnc:
        y = fnc(x)
        plt.plot(x, y, 'r', label= 'fnc')
    plt.plot(xinter, yinter, 'ok')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(label)
    plt.legend()
    plt.grid()

    plt.show()

	# adjust return for nodes, values and weights
    return

# ###########
# # test
# ###########


# n=12

# xi = np.linspace(-1,1,n)
# yi = f(xi)

# x_ch = chebyshev_pts(n)
# y_ch = f(x_ch)

# # plag = lagrange(xi, yi, f)
# # plag = lagrange(x_ch, y_ch, f)

# def f(x):
#     f_ = np.exp(-25*x**2)
#     return f_


# pfh = floater_hormann(4, xi, yi, f)
# pfh = floater_hormann(6, xi, yi, f)
# pfh = floater_hormann(8, xi, yi, f)






# %%
