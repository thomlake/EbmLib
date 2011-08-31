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
#	backproptrainer.py
# description:
#	Backpropagation for Autoencoders
#---------------------------------------#

import numpy as np
from .. units import derivatives

class BackPropTrainer(object):
	"""backpropagation trainer class

	:param net: the model to train
	:param lr: learning rate
	:param m: momentum
	:param l2: l2 regularization penalty

	:type net: ebmlib.autoencoder.AutoEncoder
	:type lr: float
	:type m: float
	:type l2: float
	"""
	def __init__(self, net, lr = 0.1, m = 0.9, l2 = 0.0001):
		self.lr, self.m, self.l2 = lr, m, l2
		self.dherr = derivatives[net.htype]
		self.doerr = derivatives[net.otype]

	def learn(self, net, x, m = True, l2 = True):
		"""weight update for single training example

		:param net: model to update
		:param x: training example
		:param m: include momentum term in cost function
		:param l2: include l2 regularization term in cost function
		:type net: ebmlib.autoencoder.AutoEncoder
		:type x: numpy.array
		:type m: bool
		:type l2: bool
		:rtype: None
		"""
		net.ff(x)
		eo = self.doerr(net.o) * (x - net.o)
		eh = self.dherr(net.h) * np.dot(net.woh.T, eo)

		if l2:
			dwoh = self.lr * (np.outer(eo, net.h) - self.l2 * net.woh)
			dwhi = self.lr * (np.outer(eh, x) - self.l2 * net.whi)
		else:
			dwoh = self.lr * np.outer(eo, net.h)
			dwhi = self.lr * np.outer(eh, x)
		
		dob = self.lr * eo
		dhb = self.lr * eh

		if m:
			dwoh += self.m * net.dwoh
			dwhi += self.m * net.dwhi
			dob += self.m * net.dob
			dhb += self.m * net.dhb

		net.woh += dwoh
		net.whi += dwhi
		net.ob += dob
		net.hb += dhb

		net.dwoh = dwoh
		net.dwhi = dwhi
		net.dob = dob
		net.dhb = dhb

	def batchlearn(self, net, X, m = True, l2 = True):
		"""weight update for a batch of training examples

		:param net: model to update
		:param X: examples
		:param m: include momentum term in cost function
		:param l2: include l2 regularization term in cost function

		:type net: ebmlib.autoencoder.AutoEncoder
		:type X: 2d numpy.array or list of numpy.array
		:type m: bool
		:type l2: bool

		:rtype: None
		"""
		dwoh = np.zeros(net.woh.shape)
		dwhi = np.zeros(net.whi.shape)
		dob = np.zeros(net.ob.shape)
		dhb = np.zeros(net.hb.shape)
		
		for x in X:
			net.ff(x)
			eo = x - net.o
			eh = self.dherr(net.h) * np.dot(net.woh.T, eo)
			dwoh += np.outer(eo, net.h)
			dwhi += np.outer(eh, x)
			dob += eo
			dhb += eh

		if l2:
			dwoh = self.lr * (dwoh - self.l2 * net.woh)/len(X)
			dwhi = self.lr * (dwhi - self.l2 * net.whi)/len(X)
		else:
			dwoh = self.lr * dwoh / len(X)
			dwhi = self.lr * dwhi / len(X)

		dob = self.lr * dob / len(X)
		dhb = self.lr * dhb / len(X)
		
		if m:
			dwoh += self.m * net.dwoh
			dwhi += self.m * net.dwhi
			dob += self.m * net.dob
			dhb += self.m * net.dhb

		net.woh += dwoh
		net.whi += dwhi
		net.ob += dob
		net.hb += dhb

		net.dwoh = dwoh
		net.dwhi = dwhi
		net.dob = dob
		net.dhb = dhb

class SparseBackPropTrainer(object):
	"""backpropagation with sparisty constraint trainer class

	:param net: the model to train
	:param lr: learning rate
	:param m: momentum
	:param l2: l2 regularization penalty
	:param spen: sparisty penaly
	:param p: desired sparsity
	:param pdecay: decay rate for mean approximation

	:type net: ebmlib.autoencoder.AutoEncoder
	:type lr: float
	:type m: float
	:type l2: float
	:type spen: float
	:type p: float
	:type pdecay: float
	"""
	def __init__(self, net, lr = 0.1, m = 0.9, l2 = 0.0001, p = 0.1, spen = 0.0001, pdecay = 0.9):
		self.lr, self.m, self.l2, self.p, self.spen, self.pdecay = lr, m, l2, p, spen, pdecay
		self.dherr = derivatives[net.htype]
		self.doerr = derivatives[net.otype]

		self.q = np.zeros(net.nhid)

	def sparseterm(self, h):
		"""compute the sparse penalty term and update the exponential decaying mean approximation

		:param h: hidden state
		:type h: numpy.array

		:rtype: float = spen*(q-p)
		"""
		qnew = (self.pdecay * self.q) + ((1 - self.pdecay) * h)
		self.q = qnew
		return self.spen * (qnew - self.p)

	def batchsparseterm(self, q):
		"""compute the sparsity penalty
	
		:param q: mean unit activities
		:type q: numpy.array

		:rtype: float = spen*(q-p)
		"""
		return self.spen * (q - self.p)

	def learn(self, net, x, m = True, l2 = True):
		"""weight update for single training example

		:param net: model to update
		:param x: training example
		:param m: include momentum term in cost function
		:param l2: include l2 regularization term in cost function

		:type net: ebmlib.autoencoder.AutoEncoder
		:type x: numpy.array
		:type m: bool
		:type l2: bool

		:rtype: None
		"""
		net.ff(x)
		eo = self.doerr(net.o) * (x - net.o)
		eh = self.dherr(net.h) * (np.dot(net.woh.T, eo) - self.sparseterm(net.h))

		if l2:
			dwoh = self.lr * (np.outer(eo, net.h) - self.l2 * net.woh)
			dwhi = self.lr * (np.outer(eh, x) - self.l2 * net.whi)
		else:
			dwoh = self.lr * np.outer(eo, net.h)
			dwhi = self.lr * np.outer(eh, x)
		
		dob = self.lr * eo
		dhb = self.lr * eh

		if m:
			dwoh += self.m * net.dwoh
			dwhi += self.m * net.dwhi
			dob += self.m * net.dob
			dhb += self.m * net.dhb

		net.woh += dwoh
		net.whi += dwhi
		net.ob += dob
		net.hb += dhb

		net.dwoh = dwoh
		net.dwhi = dwhi
		net.dob = dob
		net.dhb = dhb

	def batchlearn(self, net, X, m = True, l2 = True):
		"""weight update for a batch of training examples

		:param net: model to update
		:param x: example
		:param m: include momentum term in cost function
		:param l2: include l2 regularization term in cost function

		:type rbm: ebmlib.autoencoder.AutoEncoder
		:type X: 2d numpy.array of list of numpy.array
		:type m: bool
		:type l2: bool

		:rtype: None
		"""
		dwoh = np.zeros(net.woh.shape)
		dwhi = np.zeros(net.whi.shape)
		dob = np.zeros(net.ob.shape)
		dhb = np.zeros(net.hb.shape)
		
		phat = np.zeros(net.nhid)
		for x in X:
			net.ff(x)
			phat += net.h
		phat /= len(X)
		sparse_penalty_term = self.batchsparseterm(phat)
		for x in X:
			net.ff(x)
			eo = x - net.o
			eh = self.dherr(net.h) * (np.dot(net.woh.T, eo) - sparse_penalty_term)
			dwoh += np.outer(eo, net.h)
			dwhi += np.outer(eh, x)
			dob += eo
			dhb += eh

		if l2:
			dwoh = self.lr * (dwoh - self.l2 * net.woh)/len(X)
			dwhi = self.lr * (dwhi - self.l2 * net.whi)/len(X)
		else:
			dwoh = self.lr * dwoh / len(X)
			dwhi = self.lr * dwhi / len(X)

		dob = self.lr * dob / len(X)
		dhb = self.lr * dhb / len(X)
		
		if m:
			dwoh += self.m * net.dwoh
			dwhi += self.m * net.dwhi
			dob += self.m * net.dob
			dhb += self.m * net.dhb

		net.woh += dwoh
		net.whi += dwhi
		net.ob += dob
		net.hb += dhb

		net.dwoh = dwoh
		net.dwhi = dwhi
		net.dob = dob
		net.dhb = dhb
