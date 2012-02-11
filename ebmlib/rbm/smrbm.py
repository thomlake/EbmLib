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
#	Restricted Boltzmann Machine class
#---------------------------------------#

import numpy as np
from .. units import unittypes, sigmoid, rthresh, pthresh, softmax

class SoftmaxRbm(object):
	"""restricted boltzmann machine class

	:param nvis: number of visible units
	:param nhid: number of hidden units
	:param vtype: visible unit type, see units.py for available types
	:param htype: hidden unit type, see units.py for available types
	:type nvis: int
	:type nhid: int
	:type vtype: string
	:type htype: string
	"""
	def __init__(self, nvis, nhid):
		self.nvis = nvis
		self.nhid = nhid
		# weights
		self.W = np.random.uniform(low = -0.2, high = 0.2, size = (nhid, nvis))
		# biases
		self.vb = np.zeros(nvis)
		self.hb = np.zeros(nhid)
		# deltas
		self.dW = np.zeros((nhid, nvis))
		self.dvb = np.zeros(nvis)
		self.dhb = np.zeros(nhid)

	def ff(self, v):
		"""sample hidden given visible

		:param v: visible unit state
		:type v: numpy.array
		:returns: hidden state
		:rtype: numpy.array
		"""
		return softmax(np.dot(self.W, v) + self.hb)

	def fb(self, h):
		"""sample visible given hidden

		:param h: hidden unit state
		:type h: numpy.array
		:returns: visible state
		:rtype: numpy.ndarray
		"""
		return sigmoid(np.dot(self.W.T, h) + self.vb)

	def hid_sample(self, h, det = False, index = False):
		if det:
			if index:
				return h.argmax()
			s = np.zeros(self.nhid)
			s[h.argmax()] = 1
			return s
		else:
			if index:
				return h.cumsum().searchsorted(np.random.random())
			s = np.zeros(self.nhid)
			s[h.cumsum().searchsorted(np.random.random())] = 1
			return s

	def vis_sample(self, v):
		return rthresh(v)

	def reconstruct(self, v):
		return rthresh(self.fb(self.hid_sample(self.ff(v))))
	
	def free_energy(self, v):
		"""compute the free energy of a visible vector

		:param v: visible unit state
		:type v: numpy.ndarray
		:returns: free energy of v
		:rtype: float 
		"""
		vbias_term = -1 *np.sum(v * self.vb)
		hidden_term = -1 * np.sum(np.log(1 + np.exp(np.dot(self.W, v) + self.hb)))
		return vbias_term + hidden_term

	def energy(self, v, h):
		vbias_term = -1 * np.sum(v * self.vb)
		hbias_term = -1 * np.sum(h * self.hb)
		vhterm = -1 * np.sum(self.W * np.outer(h, v))
		return vbias_term + hbias_term + vhterm

	def __getstate__(self):
		d = {
			'nvis':		self.nvis,
			'nhid':		self.nhid,
			'W':		self.W.copy(),
			'vb':		self.vb.copy(),
			'hb':		self.hb.copy(),
			'dW':		self.dW.copy(),
			'dhb':		self.dhb.copy(),
			'dvb':		self.dvb.copy()}
		return d

	def __setstate__(self, d):
		self.nvis = 	d['nvis']
		self.nhid =		d['nhid']
		self.W = 		d['W']
		self.vb = 		d['vb']
		self.hb =		d['hb']
		self.dW = 		d['dW']
		self.dvb =		d['dvb']
		self.dhb =		d['dhb']

