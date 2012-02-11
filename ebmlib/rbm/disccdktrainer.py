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
#	Contrastive Divergence for training DRBMs and Sparse DRBMs
#---------------------------------------#

import numpy as np
from .. units import sigmoid

class DiscCdkTrainer(object):
	"""contrastive divergence trainer class

	:param rbm: the model to train
	:param lr: learning rate
	:param m: momentum
	:param l2: l2 regularization penalty
	:param spen: sparisty penaly
	:param p: desired sparsity
	:param pdecay: decay rate for mean approximation

	:type rbm: ebmlib.rbm.Drbm
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

	def cross_entropy(self, rbm, x):
		"""compute the cross entropy of a reconstruction of an input x"""
		v = rbm.fb(rbm.ff(x))
		return (x * np.log(v + 1e-8) + (1 - x) * np.log(1 - v + 1e-8)).sum()

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


	def oyj(self, rbm, y, j, x):
		cj = rbm.hb[j]
		sumW = (rbm.Whv[j] * x).sum()
		Ujy = rbm.Who[j, y]
		return cj + sumW + Ujy

	def iterdisclearn(self, rbm, x, y, m = True, l2 = True):
		pclass = rbm.pclass(x)
		pos = sigmoid(np.dot(rbm.Whv, x) + np.dot(rbm.Who, y) + rbm.hb)
		neg = sigmoid(np.dot(rbm.Whv, x) + np.dot(rbm.Who, pclass) + rbm.hb)
		
		gv = np.outer(pos, x) - np.outer(neg, x)
		go = np.outer(pos, y) - np.outer(neg, pclass)

		rbm.Who += self.lr * go
		rbm.Whv += self.lr * gv


	def disclearn(self, rbm, x, y, m = True, l2 = True):
		"""sgd weight update for single visible output configuration

		:param rbm: model to update
		:param x: data vector
		:param y: 1 of k class vector
		:param m: include momentum term in cost function
		:param l2: include l2 regularization term in cost function

		:type rbm: ebmlib.rbm.Drbm
		:type x: numpy.array
		:type y: numpy.array
		:type m: bool
		:type l2: bool

		:rtype: None
		"""
		ystar = [np.zeros(rbm.nout) for i in range(rbm.nout)]
		for i, v in enumerate(ystar):
			v[i] = 1

		pos = sigmoid(np.dot(rbm.Whv, x) + np.dot(rbm.Who, y) + rbm.hb)
		pclass = rbm.pclass(x)
		negs = [pclass[i] * sigmoid(np.dot(rbm.Whv, x) + np.dot(rbm.Who, ys) + rbm.hb) for i, ys in enumerate(ystar)]
		
		posv = np.outer(pos, x)
		poso = np.outer(pos, y)
		#posWhv = np.array([pos[i] * x for i in range(rbm.nhid)])
		#posWho = np.array([pos[i] * y for i in range(rbm.nhid)])

		negv = np.zeros((rbm.nhid, rbm.nvis))
		nego = np.zeros((rbm.nhid, rbm.nout))

		for i in range(rbm.nout):
			negv += np.outer(negs[i], x)
			nego += np.outer(negs[i], ystar[i])

		#negWhv = np.array([pclass[i] * negs[i] * x for i in range(rbm.nhid)])
		#negWho = np.array([pclass[i] * negs[i] for i in range(rbm.nhid)])

		gWhv = posv - negv
		gWho = poso - nego

		#gWhv = np.outer(pos, x) - np.outer(neg, x)
		#gWho = np.outer(pos, y) - np.outer(neg, pclass)
		#ghb = (pos - neg)
		#gvb = x.copy()
		#gob = y.copy()

		if l2:
			gWhv -= self.l2 * rbm.Whv
			gWho -= self.l2 * rbm.Who
			ghb -= self.l2 * rbm.hb
			gvb -= self.l2 * rbm.vb
			gob -= self.l2 * rbm.ob

		dWhv = self.lr * gWhv
		dWho = self.lr * gWho
		#dhb = self.lr * ghb
		#dvb = self.lr * gvb
		#dob = self.lr * gob

		if m:
			dWho += self.m * rbm.dWho
			dWhv += self.m * rbm.dWhv
			dhb += self.m * rbm.dhb
			dvb += self.m * rbm.dvb
			dob += self.m * rbm.dob

		rbm.Whv += dWhv
		rbm.Who += dWho
		#rbm.hb += dhb
		#rbm.vb += dvb
		#rbm.ob += dob

		rbm.dWhv = dWhv
		rbm.dWho = dWho
		#rbm.dhb = dhb
		#rbm.dvb = dvb
		#rbm.dob = dob

	def genlearn(self, rbm, x, y, k = 1, m = True, l2 = True, s = True):
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
		po = y
		ph = rbm.ff(pv, po)
		if k == 1:
			nv, no = rbm.fb(ph)
			#no = rbm.output(x)
			nh = rbm.ff(nv, no)
		else:
			nh = ph.copy()
			for i in range(k):
				nv, no = rbm.fb(nh)
				nh = rbm.ff(pv, no)

		p_hv = np.outer(ph, pv)
		p_ho = np.outer(ph, po)
		n_hv = np.outer(nh, nv)
		n_ho = np.outer(ph, no)

		gWhv = p_hv - n_hv
		gWho = p_ho - n_ho
		gob = po - no
		gvb = pv - nv
		ghb = ph - nh
		
		# regulization
		if l2:
			gWhv -= (self.l2 * rbm.Whv)
			gWho -= (self.l2 * rbm.Who)
		# sparisty
		if s:
			sparse_penalty_term = self.sparseterm(ph)
			gWhv = (gWhv.T - sparse_penalty_term).T
			gWho = (gWho.T - sparse_penalty_term).T
			ghb -= sparse_penalty_term

		dWhv = self.lr * gWhv
		dWho = self.lr * gWho
		dob = self.lr * gob
		dvb = self.lr * gvb
		dhb = self.lr * ghb

		# momentum
		if m:
			dWhv += self.m * rbm.dWhv
			dWho += self.m * rbm.dWho
			dvb += self.m * rbm.dvb
			dhb += self.m * rbm.dhb
			dob += self.m * rbm.dob

		rbm.Whv += dWhv
		rbm.Who += dWho
		rbm.vb += dvb
		rbm.hb += dhb
		rbm.ob += dob

		rbm.dWhv = dWhv
		rbm.dWho = dWho
		rbm.dvb = dvb
		rbm.dhb = dhb
		rbm.dob = dob

	def batchlearn(self, rbm, X, Y, k = 1, m = True, l2 = True, s = True):
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
		dWhv = np.zeros(rbm.Whv.shape)
		dWho = np.zeros(rbm.Who.shape)
		dvb = np.zeros(rbm.vb.shape)
		dhb = np.zeros(rbm.hb.shape)
		dob = np.zeros(rbm.ob.shape)
		
		q = np.zeros(rbm.nhid)

		for x, y in zip(X, Y):
			pv = x
			po = y
			ph = rbm.ff(pv, po)
			
			if k == 1:
				nv, no = rbm.fb(ph)
				#no = rbm.output(x)
				nh = rbm.ff(nv, no)
			else:
				nh = ph.copy()
				for i in range(k):
					nv, no = rbm.fb(nh)
					nh = rbm.ff(nv, no)
			
			p_hv = np.outer(ph, pv)
			p_ho = np.outer(ph, po)
			n_hv = np.outer(nh, nv)
			n_ho = np.outer(nh, no)
			
			dvb += pv - nv
			dhb += ph - nh
			dob += po - no
			dWhv += p_hv - n_hv
			dWho += p_ho - n_ho
			
			if s:
				q += ph
	
		dWhv /= len(X)
		dWho /= len(X)
		dhb /= len(X)
		dvb /= len(X)
		dob /= len(X)
		# regularization
		if l2:
			dWhv -= self.l2 * rbm.Whv
			dWho -= self.l2 * rbm.Who
		# sparsity
		if s:
			sparse_penalty_term = self.batchsparseterm(q/len(X))
			dWhv = (dWhv.T - sparse_penalty_term).T
			dWho = (dWho.T - sparse_penalty_term).T
			dhb -= sparse_penalty_term

			
		dWhv = self.lr * dWhv
		dWho = self.lr * dWho
		dvb = self.lr * dvb
		dhb = self.lr * dhb
		dob = self.lr * dob

		if m:
			dWhv += self.m * rbm.dWhv
			dWho += self.m * rbm.dWho
			dvb += self.m * rbm.dvb
			dhb += self.m * rbm.dhb
			dob += self.m * rbm.dob

		rbm.Whv += dWhv
		rbm.Who += dWho
		rbm.vb += dvb
		rbm.hb += dhb
		rbm.ob += dob

		rbm.dWhv = dWhv
		rbm.dWho = dWho
		rbm.dvb = dvb
		rbm.dhb = dhb
		rbm.dob = dob

