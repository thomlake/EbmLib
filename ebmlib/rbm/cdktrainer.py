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
#	cdktrainer.py
# description:
#	Contrastive Divergence for training RBMs and Sparse RBMs
#---------------------------------------#

import numpy as np

class CdkTrainer(object):
	"""contrastive divergence trainer class

	:param rbm: the model to train
	:param lr: learning rate
	:param m: momentum
	:param l2: l2 regularization penalty
	:param spen: sparisty penaly
	:param p: desired sparsity
	:param pdecay: decay rate for mean approximation

	:type rbm: ebmlib.rbm.Rbm
	:type lr: float
	:type m: float
	:type l2: float
	:type spen: float
	:type p: float
	:type pdecay: float
	"""
	def __init__(self, rbm, lr = 0.01, m = 0.9, l2 = 0.0001, 
					spen = 0.001, p = 0.1, pdecay = 0.96):
		self.lr, self.m, self.l2 = lr, m, l2
		self.spen, self.p, self.pdecay = spen, p, pdecay
		self.q = np.zeros(rbm.nhid)

	def sparseterm(self, h):
		"""compute the sparse penalty term and update the exponential decaying mean approximation

		:param h: hidden state
		:type h: numpy.array
		:returns: spen * (q - p)
		:rtype: float
		"""
		qnew = (self.pdecay * self.q) + ((1 - self.pdecay) * h)
		self.q = qnew
		return self.spen * (qnew - self.p)

	def batchsparseterm(self, q):
		"""compute the sparsity penalty
	
		:param q: mean unit activities
		:type q: numpy.array
		:returns: spen * (q - p)
		:rtype: float
		"""
		return self.spen * (q - self.p)

	def learn(self, rbm, x, k = 1, m = True, l2 = True, s = True):
		"""cdk weight update for single visible vector

		:param rbm: model to update
		:param x: data sample
		:param k: number of gibbs steps to take for negative phase
		:param m: include momentum term in cost function
		:param l2: include l2 regularization term in cost function
		:param s: include sparsity penalty term in cost function

		:type rbm: ebmlib.rbm.Rbm
		:type x: numpy.array
		:type k: int
		:type m: bool
		:type l2: bool
		:type s: bool

		:rtype: None
		"""
		pv = x
		ph = rbm.ff(x)
		if k == 1:
			nv = rbm.fb(ph)
			nh = rbm.ff(nv)
		else:
			nh = ph.copy()
			for i in range(k):
				nv = rbm.fb(nh)
				nh = rbm.ff(nv)
	
		pg = np.outer(ph, pv)
		ng = np.outer(nh, nv)

		gw = pg - ng
		gvb = pv - nv
		ghb = ph - nh

		# regulization
		if l2:
			gw -= (self.l2 * rbm.w)
		# sparisty
		if s:
			sparse_penalty_term = self.sparseterm(ph)
			gw = (gw.T - sparse_penalty_term).T
			ghb += sparse_penalty_term

		dw = self.lr * gw
		dvb = self.lr * gvb
		dhb = self.lr * ghb

		# momentum
		if m:
			dw += self.m * rbm.dw
			dvb += self.m * rbm.dvb
			dhb += self.m * rbm.dhb

		rbm.w += dw
		rbm.vb += dvb
		rbm.hb += dhb

		rbm.dw = dw
		rbm.dvb = dvb
		rbm.dhb = dhb

	def batchlearn(self, rbm, X, k = 1, m = True, l2 = True, s = True):
		"""cdk weight update for a batch visible vector

		:param rbm: model to update
		:param X: datapoints
		:param k: number of gibbs steps to take for negative phase
		:param m: include momentum term in cost function
		:param l2: include l2 regularization term in cost function
		:param s: include sparsity penalty term in cost function

		:type rbm: ebmlib.rbm.Rbm
		:type X: 2d numpy.array or list of numpy.array
		:type k: int
		:type m: bool
		:type l2: bool
		:type s: bool

		:rtype: None
		"""
		dw = np.zeros(rbm.w.shape)
		dvb = np.zeros(rbm.vb.shape)
		dhb = np.zeros(rbm.hb.shape)
		
		q = np.zeros(rbm.nhid)

		for x in X:
			pv = x
			ph = rbm.ff(x)
			if k == 1:
				nv = rbm.fb(ph)
				nh = rbm.ff(nv)
			else:
				nh = ph.copy()
				for i in range(k):
					nv = rbm.fb(nh)
					nh = rbm.ff(nv)
	
			pg = np.outer(ph, pv)
			ng = np.outer(nh, nv)
		
			dvb += (pv - nv)
			dhb += (ph - nh)
			dw += pg - ng
			
			if s:
				q += ph
	
		# regularization
		if l2:
			dw -= self.l2 * rbm.w
		# sparsity
		if s:
			sparse_penalty_term = self.batchsparseterm(q/len(X))
			dw = (dw.T - sparse_penalty_term).T
			dhb -= sparse_penalty_term

			
		dw = self.lr * dw / len(X)
		dvb = self.lr * dvb / len(X)
		dhb = self.lr * dhb  / len(X)

		if m:
			dw += self.m * rbm.dw
			dvb += self.m * rbm.dvb
			dhb += self.m * rbm.dhb

		rbm.w += dw
		rbm.vb += dvb
		rbm.hb += dhb

		rbm.dw = dw
		rbm.dvb = dvb
		rbm.dhb = dhb

