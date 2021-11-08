"""Iterpolations
* :function:`._q`
* :function:`._compute_roots`
* :function:`._mp_svd`
* :function:`._mp_qr`
* :function:`._nullspace_vector`
* :function:`._compute_roots`
* :function:`._mp_svd`
* :function:`._mp_qr`
* :function:`._nullspace_vector`
* :function:`.chebyshev_pts`
* :class:`.BarycentricRational`
* :function:`._polynomial_weights`
* :function:`.lagrange`
* :function:`.lebesgue`
* :function:`.floater-hermann`
* :function:`.aaa`
"""

# Plot
import matplotlib.pyplot as plt
# General
import math
import numpy as np
import scipy.linalg
from scipy.linalg.special_matrices import fiedler_companion

def _q(z, f, w, x):
    """Function which can compute the 'upper' or 'lower' rational function
    in a barycentric rational function.
    `x` may be a number or a column vector.
    """
    return np.sum((f * w) / (x - z), axis=-1)

def _compute_roots(w, x, use_mp):
    # Cf.:
    # Knockaert, L. (2008). A simple and accurate algorithm for barycentric
    # rational interpolation. IEEE Signal processing letters, 15, 154-157.
    if use_mp:
        from mpmath import mp

        ak = mp.matrix(w)
        ak /= sum(ak)
        bk = mp.matrix(x)

        M = mp.diag(bk)
        for i in range(M.rows):
            for j in range(M.cols):
                M[i,j] -= ak[i]*bk[j]
        lam = np.array(mp.eig(M, left=False, right=False))
        # remove one simple root
        lam = np.delete(lam, np.argmin(abs(lam)))
        return lam
    else:
        # the same procedure in standard double precision
        ak = w / w.sum()
        M = np.diag(x) - np.outer(ak, x)
        lam = scipy.linalg.eigvals(M)
        # remove one simple root
        lam = np.delete(lam, np.argmin(abs(lam)))
        return np.real_if_close(lam)

# SVD decomposiition
def _mp_svd(A, full_matrices=True):
    """Convenience wrapper for mpmath high-precision SVD."""
    from mpmath import mp
    AA = mp.matrix(A.tolist())
    U, Sigma, VT = mp.svd(AA, full_matrices=full_matrices)
    return np.array(U.tolist()), np.array(Sigma.tolist()).ravel(), np.array(VT.tolist())

# QR decomposiition
def _mp_qr(A):
    """Convenience wrapper for mpmath high-precision QR decomposition."""
    from mpmath import mp
    AA = mp.matrix(A.tolist())
    Q, R = mp.qr(AA, mode='full')
    return np.array(Q.tolist()), np.array(R.tolist())

# Learn attribute attribute conj() from: scipy.linalg.qr / mp.qr
def _nullspace_vector(A, use_mp=False):
    if A.shape[0] == 0:
        # some LAPACK implementations have trouble with size 0 matrices
        result = np.zeros(A.shape[1])
        result[0] = 1.0
        if use_mp:
            from mpmath import mpf
            result = np.vectorize(mpf)(result)
        return result

    if use_mp:
        Q, _ = _mp_qr(A.T)
    else:
        Q, _ = scipy.linalg.qr(A.T, mode='full')
    return Q[:, -1].conj()

def chebyshev_pts(n):

	pi = math.pi
	tt = np.linspace(0,pi,n)
	zz = np.exp(complex(0, 1)*tt)
	x = [ele.real for ele in zz]
	return np.array(x)

def _polynomial_weights(x):
    n = len(x)
    return np.array([
            1.0 / np.prod([x[i] - x[j] for j in range(n) if j != i])
            for i in range(n)
    ])

class BarycentricRational:
    """
    source: https://github.com/c-f-h/baryrat/blob/d6741e410097b6a84f2a050ae36896decaddb1c1/baryrat.py#L541
    A class representing a rational function in barycentric representation.
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

    def make_plot(self, f, name):
        """Plot
        - the function
        - rational function at all points of in the interval [min(xj), max(xj)].
        - and the chosen interpolation points
        """
        # contruct title fro; info
        xj = self.nodes

        x = np.linspace(xj.min(), xj.max(), int(xj.max()-xj.min())*100 )
        r_y = self.__call__(x)
        f_y = f(x)

        plt.plot(x, r_y, label= 'interpolation')
        plt.title('{} with {} points'.format(name, len(self.nodes)))

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.savefig('{}_{}_pts.png'.format(name, len(self.nodes)))
        plt.show()

    def numerator(self):
        """Return a new :class:`BarycentricRational` which represents the numerator polynomial."""
        weights = _polynomial_weights(self.nodes)
        return BarycentricRational(self.nodes.copy(), self.values * self.weights / weights, weights)

    def denominator(self):
        """Return a new :class:`BarycentricRational` which represents the denominator polynomial."""
        weights = _polynomial_weights(self.nodes)
        return BarycentricRational(self.nodes.copy(), self.weights / weights, weights)

    def degree_numer(self, tol=1e-12):
        """Compute the true degree of the numerator polynomial.
        Uses a result from [Berrut, Mittelmann 1997].
        """
        N = len(self.nodes) - 1
        for defect in range(N):
            if abs(np.sum(self.weights * self.weights * (self.nodes ** defect))) > tol:
                return N - defect
        return 0

    def degree_denom(self, tol=1e-12):
        """Compute the true degree of the denominator polynomial.
        Uses a result from [Berrut, Mittelmann 1997].
        """
        N = len(self.nodes) - 1
        for defect in range(N):
            if abs(np.sum(self.weights * (self.nodes ** defect))) > tol:
                return N - defect
        return 0

    def degree(self, tol=1e-12):
        """Compute the pair `(m,n)` of true degrees of the numerator and denominator."""
        return (self.degree_numer(tol=tol), self.degree_denom(tol=tol))
    # empty
    def eval_deriv(self, x, k=1):
        """Evaluate the `k`-th derivative of this rational function at a scalar
        node `x`, or at each point of an array `x`. Only the cases `k <= 2` are
        currently implemented.
        Note that this function may incur significant numerical error if `x` is
        very close (but not exactly equal) to a node of the barycentric
        rational function.
        References:
            https://doi.org/10.1090/S0025-5718-1986-0842136-8 (C. Schneider and
            W. Werner, 1986)
        """
        pass
    # empty
    def gain(self):
        """The gain in a poles-zeros-gain representation of the rational function,
        or equivalently, the value at infinity.
        """
        pass
    # empty
    def reduce_order(self):
        """Return a new :class:`BarycentricRational` which represents the same rational
        function as this one, but with minimal possible order.
        See (Ionita 2013), PhD thesis.
        """
        pass
    # empty
    def polres(self, use_mp=False):
        """Return the poles and residues of the rational function.
        If ``use_mp`` is ``True``, uses the ``mpmath`` package to compute the
        result. Set `mpmath.mp.dps` to the desired number of decimal digits
        before use.
        """
        pass
    # empty
    def poles(self, use_mp=False):
        """Return the poles of the rational function.
        If ``use_mp`` is ``True``, uses the ``mpmath`` package to compute the
        result. Set `mpmath.mp.dps` to the desired number of decimal digits
        before use.
        """
        pass
    # empty
    def jacobians(self, x):
        """Compute the Jacobians of `r(x)`, where `x` may be a vector of
        evaluation points, with respect to the node, value, and weight vectors.
        The evaluation points `x` may not lie on any of the barycentric nodes
        (unimplemented).
        Returns:
            A triple of arrays with as many rows as `x` has entries and as many
            columns as the barycentric function has nodes, representing the
            Jacobians with respect to :attr:`self.nodes`, :attr:`self.values`,
            and :attr:`self.weights`, respectively.
        """
        pass

# lebesgue constant varying on points dist.
def plot_lebesgue(xinter):
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

# basic lagrange interpolation
def lagrange(xinter, fnc):
    """
    Summary:    From the intersection coordinates (xinter, yinter), thins function generates
    --------    and plots the lagrange interpolation curve based on the .
    Input:      - xinter: x intercept (i.e. chebyshev or equi-dist)
    ------      - fnc: xinter correspondent points in domain y or the fnc
    Output:     Compute the interpolating polynomial for the given nodes and values in
    -------     barycentric form.
    """
    if callable(fnc):
        f = fnc(xinter)
        yinter = np.array(f)
    else:
        assert len(xinter) == len(fnc), "the x and y intersections coordinates have different length"
        yinter = fnc

    n = len(xinter)
    if n != len(yinter):
        raise ValueError('input arrays should have the same length')
    weights = _polynomial_weights(xinter)

    return BarycentricRational(xinter, yinter, weights)

# rational window interpolation
def floater_hormann(d, xinter, fnc):
    """
    Summary:    From the intersection coordinates (xinter, yinter), thins function generates
    --------    and plots using Floater-Hormann rational interpolant curve based on the .
    Input:      - d: the window size
    ------      - xinter: x intercept (i.e. chebyshev or equi-dist)
                - fnc: case the funciont is known
    Output:     Compute the interpolating polynomial for the given nodes and values in
    -------     barycentric form.
    """

    # Assertions
    if callable(fnc):
        f = fnc(xinter)
        yinter = np.array(f)
    else:
        assert len(xinter) == len(fnc), "the x and y intersections coordinates have different length"
        yinter = fnc
    if len(yinter) != len(xinter):
        raise ValueError('input arrays should have the same length')
    if not (0 <= d <= len(xinter)-1):
        raise ValueError('window parameter should be between 0 and n')


    weights = np.zeros(len(xinter))

    for i in range(len(xinter)):
        Ji = range(max(0, i-d), min(i, len(xinter)-1-d) + 1)
        w = 0.0
        for k in Ji:
            w += np.prod([1.0 / abs(xinter[i] - xinter[j]) for j in range(k, k+d+1) if j != i])
        weights[i] = (-1.0)**(i-d) * w

    return BarycentricRational(xinter, yinter, weights)

# best rational approximation
def aaa(xinter, fnc=None, tol=1e-13, mmax=100,  return_errors=False):

    """
    Summary:    Greedy algorithm tha will find the optimum weight for a barycentric interpolation
    --------    given the number of interactions and the error rate tolerance.

    Input:      - xinter: intersection knots (np.array)
    ------      - yinter or fnc : intersection knots values in respect to fnc (np.array) or
    the function to be interpolated (function)
                - mmax: max number of interpolation points up to the tolarance (int)
                - tol: tolerance rate to drop off the interaction loop

    Output:     - xj: optimum knots
    -------     - yj: optimum knots values
                - wj: the weight values for the barycentric interpolation form
                - errors : residual values
    """
    # Assertions
    if callable(fnc):
        f = fnc(xinter)
        yinter = np.array(f)
    else:
        assert len(xinter) == len(fnc), "the x and y intersections coordinates have different length"
        yinter = fnc

    # Initiations :
    J = list(range(len(yinter)))                # array of sequence size yinter
    xj = np.empty(0, dtype=xinter.dtype)        # empty array xj
    yj = np.empty(0, dtype=yinter.dtype)        # empty array yj
    C = []                                      # cauchy matrix
    errors = []                                 # error matrix

    # Loop criterions
    reltol = tol*np.linalg.norm(yinter, np.inf)      # infinite norm, the results would be much the same with inf points.
    R = np.mean(yinter) * np.ones_like(yinter)  # yint mean array of size len(yint) -> start point

    for m in range(mmax):
        # find largest residual
        jj = np.argmax(abs(yinter - R))
        xj = np.append(xj, (xinter[jj],))
        yj = np.append(yj, (yinter[jj],))
        J.remove(jj)
        # Cauchy matrix containing the basis functions as columns
        C = 1.0 / (xinter[J,None] - xj[None,:])
        # Loewner matrix
        A = (yinter[J,None] - yj[None,:]) * C
        # compute weights as right singular vector for smallest singular value
        _, _, Vh = np.linalg.svd(A)
        wj = Vh[-1, :].conj()
        # approximation: numerator / denominator
        N = C.dot(wj * yj)
        D = C.dot(wj)

        # update residual
        R = yinter.copy()
        R[J] = N / D

        # check for convergence
        errors.append(np.linalg.norm(yinter - R, np.inf))
        if errors[-1] <= reltol:
            break
    r = BarycentricRational(xj, yj, wj)
    return (r, errors) if return_errors else r

def f(x):
    f_ = np.exp(-25*x**2)
    return f_

if __name__ == '__main__':
    print("hello")
    # number of nodes and distribution
    n = 12
    nodes = np.linspace(-1,1,n)
    values = f(nodes)

    # Lagrange interpolations
    lg = lagrange(nodes, f)
    lg.make_plot(f,'Lagrange')

    # AAA interpolations
    aaa_ = aaa(nodes, fnc=f, tol=1e-13, mmax=100,  return_errors=False)
    aaa_.make_plot(f, 'AAA')

    # Floater Hormann interpolations
    fh = floater_hormann(4, nodes, f)
    fh.make_plot(f, 'Floater-Hormann_(d={})'.format(4))
