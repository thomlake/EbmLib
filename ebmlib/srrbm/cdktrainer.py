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
#	Contrastive Divergence for training recursive RBM variants
#---------------------------------------#
from .. units import rthresh, sigmoid
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

	:type rbm: ebmlib.srrbm.Srrbm
	:type lr: float
	:type m: float
	:type l2: float
	:type spen: float
	:type p: float
	:type pdecay: float
	"""
	def __init__(self, rbm, lr = 0.01, m = 0.4, l2 = 0.001, 
					spen = 0.001, p = 0.1, pdecay = 0.96):
		self.lr, self.m, self.l2 = lr, m, l2
		self.spen, self.p, self.pdecay = spen, p, pdecay
		self.q = np.zeros(rbm.nhid)

	def cross_entropy(self, x, v):
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
		:type rbm: ebmlib.srrbm.Srrbm
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
			#nv, nc = rbm.fb(rthresh(ph))
			nv, nc = rbm.fb(rbm.hid_sample(ph))
			nh = rbm.ff(rbm.vis_sample(nv), rbm.hid_sample(nc))
		else:
			nh = ph.copy()
			for i in range(k):
				nv, nc = rbm.fb(rbm.hid_sample(nh))
				nh = rbm.ff(rbm.vis_sample(nv), rbm.hid_sample(nc))
	
		pgv = np.outer(ph, pv)
		pgc = np.outer(ph, pc)
		ngv = np.outer(nh, nv)
		ngc = np.outer(nh, nc)

		gWhv = pgv - ngv
		gWhc = pgc - ngc
		
		gvb = pv - nv
		gcb = pc - nc
		ghb = ph - nh

		# regulization
		if l2:
			gWhv -= self.l2 * rbm.Whv
			gWhc -= self.l2 * rbm.Whc
		# sparsity
		if s:
			sparse_penalty_term = self.sparseterm(ph)
			gWhv = (gWhv.T - sparse_penalty_term).T
			gWhc = (gWhc.T - sparse_penalty_term).T
			ghb -= sparse_penalty_term
			gcb -= sparse_penalty_term

		dWhv = self.lr * gWhv
		dWhc = self.lr * gWhc
		dvb = self.lr * gvb
		dcb = self.lr * gcb
		dhb = self.lr * ghb

		# momentum
		if m:
			dWhv += self.m * rbm.dWhv
			dWhc += self.m * rbm.dWhc
			dvb += self.m * rbm.dvb
			dcb += self.m * rbm.dcb
			dhb += self.m * rbm.dhb

		rbm.Whv += dWhv
		rbm.Whc += dWhc
		rbm.vb += dvb
		rbm.cb += dcb
		rbm.hb += dhb

		rbm.dWhv = dWhv
		rbm.dWhc = dWhc
		rbm.dvb = dvb
		rbm.dcb = dcb
		rbm.dhb = dhb

		rbm.h = ph
		#rbm.push(x)

	def batchlearn(self, rbm, X, k = 1, m = True, l2 = True, s = True):
		"""cdk weight update for a sequence of visible vector

		:param rbm: model to update
		:param X: datapoints
		:param k: number of gibbs steps to take for negative phase
		:param m: include momentum term in cost function
		:param l2: include l2 regularization term in cost function
		:param s: include sparsity penalty term in cost function
		:type rbm: ebmlib.srrbm.Srrbm
		:type X: 2d numpy.array or list of numpy.array
		:type k: int
		:type m: bool
		:type l2: bool
		:type s: bool
		:rtype: None
		"""
		gWhv = np.zeros(rbm.Whv.shape)
		gWhc = np.zeros(rbm.Whc.shape)
		gvb = np.zeros(rbm.vb.shape)
		gcb = np.zeros(rbm.cb.shape)
		ghb = np.zeros(rbm.hb.shape)
		
		q = np.zeros(rbm.nhid)

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
					nv, nc = rbm.fb(rbm.hid_sample(nh))
					nh = rbm.ff(rbm.vis_sample(nv), rbm.hid_sample(nc))
	
			gWhv += np.outer(ph, pv) - np.outer(nh, nv)
			gWhc += np.outer(ph, pc) - np.outer(nh, nc)
		
			gvb += (pv - nv)
			gcb += (pc - nc)
			ghb += (ph - nh)

			if s:
				q += ph
			
			#rbm.push(x)
			rbm.h = ph

		gWhv /= len(X)
		gWhc /= len(X)
		gvb /= len(X)
		ghb /= len(X)
		gcb /= len(X)

		# regulization
		if l2:
			gWhv -= self.l2 * rbm.Whv
			gWhc -= self.l2 * rbm.Whc
		# sparsity
		if s:
			sparse_penalty_term = self.batchsparseterm(q/len(X))
			gWhv = (gWhv.T - sparse_penalty_term).T
			gWhc = (gWhc.T - sparse_penalty_term).T
			ghb -= sparse_penalty_term
			gcb -= sparse_penalty_term

		#dwv = self.lr / len(X) * gwv
		#dwc = self.lr / len(X) * gwc
		#dvb = self.lr / len(X) * gvb
		#dcb = self.lr / len(X) * gcb
		#dhb = self.lr / len(X) * ghb
		
		dWhv = self.lr * gWhv
		dWhc = self.lr * gWhc
		dvb = self.lr * gvb
		dcb = self.lr * gcb
		dhb = self.lr * ghb


		if m:
			dWhv += self.m * rbm.dWhv
			dWhc += self.m * rbm.dWhc
			dvb += self.m * rbm.dvb
			dcb += self.m * rbm.dcb
			dhb += self.m * rbm.dhb

		rbm.Whv += dWhv
		rbm.Whc += dWhc
		rbm.vb += dvb
		rbm.cb += dcb
		rbm.hb += dhb

		rbm.dWhv = dWhv
		rbm.dWhc = dWhc
		rbm.dvb = dvb
		rbm.dcb = dcb
		rbm.dhb = dhb


