#---------------------------------------#
#	This file is part of EbmLib.
#
#	EbmLib is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	EbmLib is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with EbmLib.  If not, see <http://www.gnu.org/licenses/>.
#---------------------------------------#
# author:
#	tllake 
# email:
#	<thomas.l.lake@wmich.edu>
#	<thom.l.lake@gmail.com>
# date:
#	2011.08.30
# file:
#	units.py
# description:
#	Several common activation functions used for energy based models.
#	The derivatives of these functions given their output.
#---------------------------------------#

import numpy

#--- SAMPLING FUNCTIONS --#
def pthresh(x):
	"""probabilistic step funtion of x

	:param x: input
	:type x: numpy.array
	:returns: 1 if sigmoid(x_i) >= ~ U(0,1) else 0 for x_i \in x
	:rtype: numpy.array
	"""
	return numpy.array(sigmoid(x) >= numpy.random.random(x.shape), dtype = numpy.float32)

def rthresh(x):
	""" sample r, r_i ~ bernoulli(p = x_i)
	:param x: input
	:type x: numpy.array
	:returns: 1 if x_i >= ~ U(0,1) else 0 for x_i \in x
	:rtype: numpy.array
	"""
	return numpy.array(x >= numpy.random.random(x.shape), dtype = numpy.float32)

def cat(x):
	""" sample categorical where p(x_i) = \frac{x_i}{\sum_{j = 1}^{n} x_j},
	
	NOTE: this faster than drawing a single multinomial sample with numpy
	:param x: input
	:type x: numpy.array
	:returns: binary vector r, where r_i = 1 if i is the chosen category else r_i = 0
	:rtype: numpy.array
	"""
	r = np.zeros(len(x))
	r[h.cumsum().searchsorted(np.random.random())] = 1
	return r

def catidx(x):
	""" sample categorical where p(x_i) = \frac{x_i}{\sum_{j = 1}^{n} x_j},
	
	NOTE: this faster than drawing a single multinomial sample with numpy
	:param x: input
	:type x: numpy.array
	:returns: the index i of the chosen category, 0 <= i < len(x)
	:rtype: int
	"""
	return x.cumsum().searchsorted(np.random.random())

#--- ACTIVATION FUNCTIONS ---#
def thresh(x, t = 0.5):
	""" step function
	:param x: input
	:param t: threshold
	:type x: numpy.array
	:type t: float
	:returns: 1 if x_i >= t else 0 for x_i \in x
	:rtype: numpy.array
	"""
	return numpy.array(x >= t, dtype = numpy.float32)

def sigmoid(x):
	"""logisitc sigmoid of x

	:param x: input
	:type x: numpy.array
	:returns: 1 / (1 + e^(-x))
	:rtype: numpy.array
	"""
	return 1./(1.+numpy.exp(-x))

def tanh(x):
	"""hyperbolic tangent of x

	:param x: input
	:type x: numpy.array
	:returns: tanh(x)
	:rtype: numpy.array
	"""
	return numpy.tanh(x)

def rectlinear(x):
	"""rectified linear function of x

	:param x: input
	:type x: numpy.array
	:returns: max[0, x + ~ N(0, 1 / (1 + (e^(-x))))]
	:rtype: numpy.array
	"""
	return numpy.maximum(0., x + sigmoid(x) * numpy.random.normal(0., 1, x.shape))

def linear(x):
	"""identity function

	:param x: input
	:type x: numpy.array
	:returns: x
	:rtype: numpy.array
	"""
	return x

def softmax(x):
	"""softmax function of x

	:param x: input
	:type x: numpy.array
	:returns: e^x_i / sum j = 1 to n e^x_j
	:rtype: numpy.array
	"""
	e_to_the_x = numpy.exp(x)
	return e_to_the_x / e_to_the_x.sum()

def detcat(x):
	"""deterministic categorical, i.e. 1 of k binary
	
	:param x: input
	:type x: numpy.array
	:returns: zero array with the max of x set to 1
	:rtype: numpy.array
	"""
	r = numpy.zeros(len(x))
	r[x.argmax()] = 1
	return r

def detcatidx(x):
	""" index of a deterministic categorical, i.e. 1 of k binary
	
	:param x: input
	:type x: numpy.array
	:returns: i = x.argmax(), 0 <= i < len(x)
	:rtype: int
	"""
	return x.argmax()

#--- DERIVATIVES ---#
def dsigmoid(y):
	"""derivative of sigmoid

	:param y: output of sigmoid
	:type y: numpy.array
	:returns: y * (1 - y)
	:rtype: numpy.array
	"""
	return y * (1 - y)

def dtanh(y):
	"""derivative of tanh

	:param y: output of tanh
	:type y: numpy.array
	:returns: 1 - y^2
	:rtype: numpy.array
	"""
	return 1 - y**2

def dlinear(y):
	"""derivative of identity

	:param y: output of identity
	:type y: numpy.array
	:returns: 1
	:rtype: float
	"""
	return 1.

def drectlinear(y):
	"""derivative of rectified linear

	:param y: output of rectified linear
	:type y: numpy.array
	:returns: 1 if y_i > 0 else 0 for y_i \in y
	:rtype: numpy.array
	"""
	return numpy.array(y > 0, dtype = numpy.float32)

# get function given the function name
unittypes = {
	'pthresh' : pthresh,
	'sigmoid': sigmoid,
	'tanh': numpy.tanh,
	'rectlinear': rectlinear,
	'linear': linear,
	'softmax': softmax,
	'cat': cat}

# get the deirvative of a function given the name
derivatives = {
	'sigmoid': dsigmoid,
	'tanh': dtanh,
	'rectlinear': drectlinear,
	'linear': dlinear,
	'softmax': dlinear} # assuming cross entropy error funtion

