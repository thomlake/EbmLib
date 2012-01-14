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
#	srrbm.py
# description:
#	Recursive Restricted Boltzmann Machine class
#---------------------------------------#

import numpy as np
from .. units import sigmoid, rthresh
from .. units import unittypes

class Srrbm(object):
	"""recursive restricted boltzmann machine class

	:param nvis: number of visible units
	:param nhid: number of hidden units
	:param vtype: visible unit type, see units.py for available types
	:param htype: hidden unit type, see units.py for available types
	:type nvis: int
	:type nhid: int
	:type vtype: string
	:type htype: string
	"""
	def __init__(self, nvis, nhid, vtype = 'pthresh', htype = 'sigmoid'):
		# unit counts
		self.nvis = nvis
		self.nhid = nhid
		# units
		self.v = np.zeros(nvis)
		self.c = np.zeros(nhid)
		self.h = np.zeros(nhid)
		# weights
		self.Whv = np.random.uniform(low = -0.2, high = 0.2, size = (nhid, nvis))
		self.Whc = np.random.uniform(low = -0.2, high = 0.2, size = (nhid, nhid))
		# biases
		self.vb = np.zeros(nvis)
		self.cb = np.zeros(nhid)
		self.hb = np.zeros(nhid)
		# delta weights
		self.dWhv = np.zeros((nhid, nvis))
		self.dWhc = np.zeros((nhid, nhid))
		# delta biases
		self.dvb = np.zeros(nvis)
		self.dcb = np.zeros(nhid)
		self.dhb = np.zeros(nhid)
		# activation functions
		self.htype = htype
		self.vtype = vtype
		self.hact = unittypes[htype]
		self.vact = unittypes[vtype]

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
		"""sample hidden given visible

		:param h: hidden unit state
		:type v: numpy.array
		:returns: visible state, context state
		:rtype: tuple (numpy.array, numpy.array)
		"""
		return sigmoid(np.dot(self.Whv.T, h) + self.vb), sigmoid(np.dot(self.Whc.T, h) + self.cb)

	def hid_sample(self, h):
		return rthresh(h)

	def vis_sample(self, v):
		return rthresh(v)

	def push(self, x):
		"""push an input x

		:param x: input
		:type x: numpy.array
		:rtype: None
		"""
		self.h = self.ff(x, self.hid_sample(self.h))

	def pop(self):
		"""pop a visible state and return it
		
		:returns: visible state
		:rtype: numpy.array
		"""
		v, self.h = self.fb(self.hid_sample(self.h))
		return self.vis_sample(v)

	def reset(self):
		"""reset the netowrks stateful hidden units to 0
		
		:rtype: None
		"""
		self.h = np.zeros(self.nhid)

	def free_energy(self, v):
		"""compute the free energy of a visible vector

		:param v: visible unit state
		:type v: numpy.ndarray
		:returns: free energy of v
		:rtype: float 
		"""
		vbias_term = -1 * np.sum(v * self.vb)
		cbias_term = -1 * np.sum(self.h * self.cb)
		hidden_term = -1 * np.sum(np.log(1 + np.exp(np.dot(self.Whv, v) + np.dot(self.Whc, self.h) + self.hb)))
		return vbias_term + cbias_term + hidden_term

	def __getstate__(self):
		d = {
			'nvis':		self.nvis,
			'nhid':		self.nhid,	
			'v':		self.v.copy(),
			'c':		self.c.copy(),
			'h':		self.h.copy(),
			'Whv':		self.Whv.copy(),
			'Whc':		self.Whc.copy(),
			'vb':		self.vb.copy(),
			'cb':		self.cb.copy(),
			'hb':		self.hb.copy(),
			'dWhv':		self.dWhv.copy(),
			'dWhc':		self.dWhc.copy(),
			'dvb':		self.dvb.copy(),
			'dcb':		self.dcb.copy(),
			'dhb':		self.dhb.copy(),
			'htype':	self.htype,
			'vtype':	self.vtype}
		return d

	def __setstate__(self, d):
		self.nvis = 	d['nvis']
		self.nhid = 	d['nhid']
		self.v = 		d['v']
		self.c = 		d['c']
		self.h = 		d['h']
		self.Whv =		d['Whv']
		self.Whc =		d['Whc']
		self.vb =		d['vb']
		self.cb =		d['cb']
		self.hb =		d['hb']
		self.dWhv =		d['dWhv']
		self.dWhc =		d['dWhc']
		self.dvb =		d['dvb']
		self.dcb =		d['dcb']
		self.dhb =		d['dhb']
		self.htype = 	d['htype']
		self.vtype = 	d['vtype']
		self.hact = unittypes[self.htype]
		self.vact = unittypes[self.vtype]

