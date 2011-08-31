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
		self.spen, self.p, self.decay = spen, p, pdecay
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
		ph = rbm.ff(pv, pc)
		if k == 1:
			nv, nc = rbm.fb(ph)
			nh = rbm.ff(nv, nc)
		else:
			nh = ph.copy()
			for i in range(k):
				nv, nc = rbm.fb(nh)
				nh = rbm.ff(nv, nc)
	
		pgv = np.outer(ph, pv)
		pgc = np.outer(ph, pc)
		ngv = np.outer(nh, nv)
		ngc = np.outer(nh, nc)

		gwv = pgv - ngv
		gwc = pgc - ngc
		
		gvb = pv - nv
		gcb = pc - nc
		ghb = ph - nh

		# regulization
		if l2:
			gwv -= self.l2 * rbm.wv
			gwc -= self.l2 * rbm.wc
		# sparsity
		if s:
			sparse_penalty_term = self.sparseterm(ph)
			gwv = (gwv.T - sparse_penalty_term).T
			gwc = (gwc.T - sparse_penalty_term).T
			ghb += sparse_penalty_term

		dwv = self.lr * gwv
		dwc = self.lr * gwc
		dvb = self.lr * gvb
		dcb = self.lr * gcb
		dhb = self.lr * ghb

		# momentum
		if m:
			dwv += self.m * rbm.dwv
			dwc += self.m * rbm.dwc
			dvb += self.m * rbm.dvb
			dcb += self.m * rbm.dcb
			dhb += self.m * rbm.dhb

		rbm.wv += dwv
		rbm.wc += dwc
		rbm.vb += dvb
		rbm.cb += dcb
		rbm.hb += dhb

		rbm.dwv = dwv
		rbm.dwc = dwc
		rbm.dvb = dvb
		rbm.dcb = dcb
		rbm.dhb = dhb

		rbm.push(x)

	def batchlearn(self, rbm, X, k = 1, m = True, l2 = True, s = True):
		"""cdk weight update for a batch visible vector

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
		gwv = np.zeros(rbm.wv.shape)
		gwc = np.zeros(rbm.wc.shape)
		gvb = np.zeros(rbm.vb.shape)
		gcb = np.zeros(rbm.cb.shape)
		ghb = np.zeros(rbm.hb.shape)
		
		q = np.zeros(rbm.nhid)

		for x in X:
			pv = x
			pc = rbm.h
			ph = rbm.ff(pv, pc)
			if k == 1:
				nv, nc = rbm.fb(ph)
				nh = rbm.ff(nv, nc)
			else:
				nh = ph.copy()
				for i in range(k):
					nv, nc = rbm.fb(nh)
					nh = rbm.ff(nv, nc)
	
			gwv += np.outer(ph, pv) - np.outer(nh, nv)
			gwc += np.outer(ph, pc) - np.outer(nh, nc)
		
			gvb += (pv - nv)
			dcb += (pc - nc)
			dhb += (ph - nh)

			if s:
				q += ph
			rbm.push(x)
		
		# regulization
		if l2:
			gwv -= self.l2 * rbm.wv
			gwc -= self.l2 * rbm.wc
		# sparsity
		if s:
			sparse_penalty_term = self.batchsparseterm(q/len(X))
			gwv = (gwv.T - sparse_penalty_term).T
			gwc = (gwc.T - sparse_penalty_term).T
			ghb -= sparse_penalty_term

		dwv = self.lr * gwv
		dwc = self.lr * gwc
		dvb = self.lr * gvb
		dcb = self.lr * gcb
		dhb = self.lr * ghb

		if m:
			dwv += self.m * rbm.dwv
			dwc += self.m * rbm.dwc
			dvb += self.m * rbm.dvb
			dcb += self.m * rbm.dcb
			dhb += self.m * rbm.dhb

		rbm.wv += dwv
		rbm.wc += dwc
		rbm.vb += dvb
		rbm.cb += dcb
		rbm.hb += dhb

		rbm.dwv = dwv
		rbm.dwc = dwc
		rbm.dvb = dvb
		rbm.dcb = dcb
		rbm.dhb = dhb

def test():
	SEQS = ['bpvpse', 'btsssxxtvve', 'btsxse', 'bpttttvve', 'bpttvve', \
				'btxxvve', 'btxse', 'bpttvpse','btssxse', 'btsxxvpxvve',	\
				'btxxtvve', 'btsssxse', 'bptvve', 'btxxvpse', 'bptttvve', \
				'bpvpse']
	moreseq = [\
				'bptvpxvpse', 'btsxxtttvve', 'bpvpxvve', 'bptvpse', \
				'btsxxtvpse', 'bptvpxvve', 'bpvve', 'btxxtvpse', \
				'bpttttvpse', 'bpvpxtttvve', 'btsxxtvpxttvpse', \
				'btxxvpxtvpse', 'btxxvpxvve', 'bptttvpxtvve', \
				'bptvpxvpxtvpse', 'btxxtttvpxtvve', 'bptttvpxvpse', \
				'btxxvpxvpse', 'btssxxvpse', 'bpvpxttvpse', 'bpvpxtvve', \
				'bptvpxvpxtvve', 'btsxxvpse', 'btssxxtvve', 'btsxxttvve', \
				'btsssxxvpse', 'bpvpxttvve', 'btsxxvve', 'btsxxvpxttvve', \
				'bptttttvve', 'btsxxtvve', 'btsssxxvve', 'btsxxtvpxvve', \
				'bpttttvpxvpse', 'btsxxttvpxtvve']
	
	ALPHABET = ['b','t','p','s','x','v','e']
	N = 20
	PATDICT = dict([(X, None) for X in ALPHABET])
	INDICT = []
	for X in ALPHABET:
		while PATDICT[X] is None:
			pat = np.zeros(N)
			pat[np.random.randint(0,N,3)] = 1.
			string_pat = ''.join(['+' if x == 1 else '-' for x in pat])
			if string_pat not in INDICT:
				PATDICT[X] = pat
				INDICT.append(string_pat)

			
	for key, val in PATDICT.items():
		print key, ''.join(['+' if x == 1. else '-' for x in val])
	
	net = Srrbm(N, 20)
	trainer = SrrbmCdkTrainer(net)
	
	epoch = 0
	while True:
		for string in SEQS:
			net.reset()
			#X = [PATDICT[char] for char in string]
			#trainer.learn_batch(net, X, s = True, l2 = False)
			for char in string:
				obs = PATDICT[char]
				trainer.learn(net, obs, s = True, l2 = False)
		epoch += 1
		if epoch and epoch % 100 == 0:
			err = 0.
			tot = 0.
			for string in SEQS:
				net.reset()
				rev = []
				for char in string:
					obs = PATDICT[char]
					rev.append(obs)
					net.push(obs)
				for char in reversed(rev):
					out = net.pop()
					err += ((out - char)**2).sum()
					tot += 1.
					#print ''.join(['+' if x == 1 else '-' for x in char]),
					#print ''.join(['+' if x == 1 else '-' for x in out])
			print err/ tot




#if __name__ == '__main__':
#	test()
