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
#	rbm.py
# description:
#	Discriminative Recursive Restricted Boltzmann Machine class
#---------------------------------------#

import numpy as np
from .. units import sigmoid, rthresh, softmax

class Drrbm(object):
	"""discriminitive recursive restricted boltzmann machine class

	:param nvis: number of visible units
	:param nhid: number of hidden units
	:type nvis: int
	:type nhid: int
	"""
	def __init__(self, nvis, nhid):
		self.nvis = nvis
		self.nhid = nhid
		# weights
		self.Whv = np.random.uniform(low = -0.2, high = 0.2, size = (nhid, nvis))
		self.Whc = np.random.uniform(low = -0.2, high = 0.2, size = (nhid, nhid))
		# biases
		self.vb = np.zeros(nvis)
		self.hb = np.zeros(nhid)
		self.cb = np.zeros(nhid)
		# deltas
		self.dWhv = np.zeros((nhid, nvis))
		self.dWhc = np.zeros((nhid, nhid))
		self.dvb = np.zeros(nvis)
		self.dhb = np.zeros(nhid)
		self.dcb = np.zeros(nhid)
		# state
		self.h = np.zeros(nhid)

	def hid_sample(self, h):
		return rthresh(h)

	def vis_sample(self, v, det = False, index = False):
		if det:
			if index:
				return v.argmax()
			s = np.zeros(self.nvis)
			s[v.argmax()] = 1
			return s
		else:
			if index:
				return v.cumsum().searchsorted(np.random.random())
			s = np.zeros(self.nvis)
			s[v.cumsum().searchsorted(np.random.random())] = 1
			return s

		r = np.zeros(self.nvis)
		r[v.argmax()] = 1
		return r

	def ff(self, v, c):
		"""sample hidden given visible and context

		:param v: visible unit state
		:param c: context unit state
		:type v: numpy.array
		:type c: numpy.array
		:returns: hidden state
		:rtype: numpy.array
		"""
		return sigmoid(np.dot(self.Whv, v) + np.dot(self.Whc, c) + self.hb)

	def fb(self, h):
		"""sample visible and context given hidden

		:param h: hidden unit state
		:type h: numpy.array
		:returns: visible and context states
		:rtype: (numpy.ndarray, numpy.ndarray)
		"""
		return softmax(np.dot(self.Whv.T, h) + self.vb), sigmoid(np.dot(self.Whc.T, h) + self.cb)

	def output(self, rtype = 'vector'):
		v_class_vecs = [np.zeros(self.nvis) for i in range(self.nvis)]
		cbias_term, h_partial = self.context_free_energy_terms(self.h)
		minfe = float('inf')
		idx = 999
		for i, v in enumerate(v_class_vecs):
			v[i] = 1
			fe = self.free_energy(v, self.h, cbias_term = cbias_term, h_partial = h_partial)
			if fe < minfe:
				minfe = fe
				idx = i
		if rtype == 'index':
			return idx
		return v_class_vecs[idx]

	def context_free_energy_terms(self, c):
		cbias_term = -1 * np.sum(c * self.cb)
		h_partial = np.dot(self.Whc, c) + self.hb
		return cbias_term, h_partial

	def free_energy(self, v, c, cbias_term = None, h_partial = None):
		"""compute the free energy of a visible and output vector

		:param v: visible unit state
		:param o: output unit state
		:type v: numpy.ndarray
		:type d: numpy.ndarray
		:returns: free energy of [v:o]
		:rtype: float 
		"""
		vbias_term = -1 * np.sum(v * self.vb)
		if cbias_term is None:
			cbias_term = -1 * np.sum(c * self.cb)
		if h_partial is None:
			h_term = -1 * np.sum(np.log(1 + np.exp(np.dot(self.Whv, v) + np.dot(self.Whc, c) + self.hb)))
		else:
			h_term = -1 * np.sum(np.log(1 + np.exp(h_partial + np.dot(self.Whv, v))))
		return vbias_term + cbias_term + h_term

	def push(self, x):
		self.h = self.ff(x, self.hid_sample(self.h))

	def pop(self):
		y, self.h = self.fb(self.hid_sample(self.h))
		return y

	def reset(self):
		self.h = np.zeros(self.nhid)

	def __getstate__(self):
		d = {
			'nvis':		self.nvis,
			'nhid':		self.nhid,
			'Whv':		self.Whv.copy(),
			'Whc':		self.Whc.copy(),
			'vb':		self.vb.copy(),
			'hb':		self.hb.copy(),
			'cb':		self.cb.copy(),
			'dWhv':		self.dWhv.copy(),
			'dWhc':		self.dWhc.copy(),
			'dhb':		self.dhb.copy(),
			'dcb':		self.dcb.copy(),
			'dvb':		self.dvb.copy()}
		return d

	def __setstate__(self, d):
		self.nvis = 	d['nvis']
		self.nhid =		d['nhid']
		self.Whv = 		d['Whv']
		self.Whc = 		d['Whc']
		self.vb = 		d['vb']
		self.hb =		d['hb']
		self.cb =		d['cb']
		self.dWhv =		d['dWhv']
		self.dWhc =		d['dWhc']
		self.dvb =		d['dvb']
		self.dhb =		d['dhb']
		self.dcb =		d['dcb']

