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
#	Contrastive Divergence for training DRRBMs and Sparse DRRBMs
#---------------------------------------#

import numpy as np

class DiscCdkTrainer(object):
	"""contrastive divergence trainer class

	:param rbm: the model to train
	:param lr: learning rate
	:param m: momentum
	:param l2: l2 regularization penalty
	:param spen: sparisty penaly
	:param p: desired sparsity
	:param pdecay: decay rate for mean approximation

	:type rbm: ebmlib.rbm.Drrbm
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

	def cross_entropy(self, x, v):
		"""compute the cross entropy of a reconstruction of an input x"""
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
		pc = rbm.h
		ph = rbm.ff(pv, rbm.hid_sample(pc))
		if k == 1:
			nv, nc = rbm.fb(rbm.hid_sample(ph))
			nh = rbm.ff(rbm.vis_sample(nv), rbm.hid_sample(nc))
		else:
			nh = ph.copy()
			for i in range(k):
				nv, nc = rbm.fb(rbm.hid_sample(nh))
				nh = rbm.ff(rbm.vis_sample(nv), rbm.hid_sample(nc))

		p_hv = np.outer(ph, pv)
		p_hc = np.outer(ph, pc)
		n_hv = np.outer(nh, nv)
		n_hc = np.outer(ph, nc)

		gWhv = p_hv - n_hv
		gWhc = p_hc - n_hc
		gcb = pc - nc
		gvb = pv - nv
		ghb = ph - nh
		
		# regulization
		if l2:
			gWhv -= (self.l2 * rbm.Whv)
			gWhc -= (self.l2 * rbm.Whc)
		# sparisty
		if s:
			sparse_penalty_term = self.sparseterm(ph)
			gWhv = (gWhv.T - sparse_penalty_term).T
			gWhc = (gWhc.T - sparse_penalty_term).T
			ghb -= sparse_penalty_term
			gcb -= sparse_penalty_term

		dWhv = self.lr * gWhv
		dWhc = self.lr * gWhc
		dcb = self.lr * gcb
		dvb = self.lr * gvb
		dhb = self.lr * ghb

		# momentum
		if m:
			dWhv += self.m * rbm.dWhv
			dWhc += self.m * rbm.dWhc
			dvb += self.m * rbm.dvb
			dhb += self.m * rbm.dhb
			dcb += self.m * rbm.dcb

		rbm.Whv += dWhv
		rbm.Whc += dWhc
		rbm.vb += dvb
		rbm.hb += dhb
		rbm.cb += dcb

		rbm.dWhv = dWhv
		rbm.dWhc = dWhc
		rbm.dvb = dvb
		rbm.dhb = dhb
		rbm.dcb = dcb

		#rbm.push(x)
		rbm.h = ph

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
		dWhv = np.zeros(rbm.Whv.shape)
		dWhc = np.zeros(rbm.Whc.shape)
		dvb = np.zeros(rbm.vb.shape)
		dhb = np.zeros(rbm.hb.shape)
		dcb = np.zeros(rbm.cb.shape)
		
		q = np.zeros(rbm.nhid)

		rbm.reset()
		for x in X:
			pv = x
			pc = rbm.h
			ph = rbm.ff(pv, rbm.hid_sample(pc))
			
			if k == 1:
				nv, nc = rbm.fb(rbm.hid_sample(ph))
				nh = rbm.ff(rbm.vis_sample(nv), rbm.hid_sample(nc))
			else:
				nh = ph.copy()
				for i in range(k):
					nv, nc = rbm.fb(rbm.hid_sample(ph))
					nh = rbm.ff(rbm.vis_sample(nv), rbm.hid_sample(nc))
			
			p_hv = np.outer(ph, pv)
			p_hc = np.outer(ph, pc)
			n_hv = np.outer(nh, nv)
			n_hc = np.outer(nh, nc)
			
			dvb += pv - nv
			dhb += ph - nh
			dcb += pc - nc
			dWhv += p_hv - n_hv
			dWhc += p_hc - n_hc
			
			if s:
				q += ph
			
			#rbm.push(x)
			rbm.h = ph
	
		dWhv /= len(X)
		dWhc /= len(X)
		dhb /= len(X)
		dvb /= len(X)
		dcb /= len(X)
		# regularization
		if l2:
			dWhv -= self.l2 * rbm.Whv
			dWhc -= self.l2 * rbm.Whc
		# sparsity
		if s:
			sparse_penalty_term = self.batchsparseterm(q/len(X))
			dWhv = (dWhv.T - sparse_penalty_term).T
			dWhc = (dWhc.T - sparse_penalty_term).T
			dhb -= sparse_penalty_term
			dcb -= sparse_penalty_term

			
		dWhv = self.lr * dWhv
		dWhc = self.lr * dWhc
		dvb = self.lr * dvb
		dhb = self.lr * dhb
		dcb = self.lr * dcb

		if m:
			dWhv += self.m * rbm.dWhv
			dWhc += self.m * rbm.dWhc
			dvb += self.m * rbm.dvb
			dhb += self.m * rbm.dhb
			dcb += self.m * rbm.dcb

		rbm.Whv += dWhv
		rbm.Whc += dWhc
		rbm.vb += dvb
		rbm.hb += dhb
		rbm.cb += dcb

		rbm.dWhv = dWhv
		rbm.dWhc = dWhc
		rbm.dvb = dvb
		rbm.dhb = dhb
		rbm.dcb = dcb

