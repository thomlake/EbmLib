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
#	Conditional Restricted Boltzmann Machine class
#---------------------------------------#

import numpy as np
from .. units import unittypes, sigmoid, pthresh, softmax

class Drbm(object):
	"""discriminitive restricted boltzmann machine class

	:param nvis: number of visible units
	:param nhid: number of hidden units
	:param vtype: visible unit type, see units.py for available types
	:param htype: hidden unit type, see units.py for available types
	:type nvis: int
	:type nhid: int
	:type vtype: string
	:type htype: string
	"""
	def __init__(self, nvis, nout, nhid, 
				vtype = 'pthresh', htype = 'pthresh', otype = 'softmax'):
		self.nvis = nvis
		self.nhid = nhid
		self.nout = nout
		# weights
		self.Whv = np.random.uniform(low = -0.2, high = 0.2, size = (nhid, nvis))
		self.Who = np.random.uniform(low = -0.2, high = 0.2, size = (nhid, nout))
		# biases
		self.vb = np.zeros(nvis)
		self.hb = np.zeros(nhid)
		self.ob = np.zeros(nout)
		# deltas
		self.dWhv = np.zeros((nhid, nvis))
		self.dWho = np.zeros((nhid, nout))
		self.dvb = np.zeros(nvis)
		self.dhb = np.zeros(nhid)
		self.dob = np.zeros(nout)
		# activation functions
		self.htype = htype
		self.vtype = vtype
		self.otype = otype
		self.hact = unittypes[htype]
		self.vact = unittypes[vtype]
		self.oact = unittypes[otype]

	def ff(self, v, o):
		"""sample hidden given visible and output

		:param v: visible unit state
		:type v: numpy.array
		:returns: hidden state
		:rtype: numpy.array
		"""
		return self.hact(np.dot(self.Whv, v) + np.dot(self.Who, o) + self.hb)

	def fb(self, h):
		"""sample visible and output given hidden

		:param h: hidden unit state
		:type h: numpy.array
		:returns: visible and output states
		:rtype: (numpy.ndarray, numpy.ndarray)
		"""
		return self.vact(np.dot(self.Whv.T, h) + self.vb), self.oact(np.dot(self.Who.T, h) + self.ob)

	def output(self, v, rtype = 'pvec'):
		p = self.pclass(v)
		if rtype == 'pvec':
			return p
		elif rtype == 'bvec':
			r = np.zeros(self.nout)
			r[p.argmax()] = 1
			return r
		elif rtype == 'index':
			return p.argmax()
		return p

	def pclass(self, v):
		p = np.zeros(self.nout)
		o_class_vecs = [np.zeros(self.nout) for i in range(self.nout)]
		vbias_term, h_partial = self.visible_free_energy_terms(v)
		for i, o in enumerate(o_class_vecs):
			o[i] = 1
			p[i] = self.free_energy(v, o, vbias_term = vbias_term, h_partial = h_partial)
		return p / p.sum()

	def visible_free_energy_terms(self, v):
		vbias_term = -1 * np.sum(v * self.vb)
		h_partial = np.dot(self.Whv, v) + self.hb
		return vbias_term, h_partial

	def free_energy(self, v, o, vbias_term = None, h_partial = None):
		"""compute the free energy of a visible and output vector

		:param v: visible unit state
		:param o: output unit state
		:type v: numpy.ndarray
		:type d: numpy.ndarray
		:returns: free energy of [v:o]
		:rtype: float 
		"""
		obias_term = -1 * np.sum(o * self.ob)
		if vbias_term is None:
			vbias_term = -1 * np.sum(v * self.vb)
		if h_partial is None:
			h_term = -1 * np.sum(np.log(1 + np.exp(np.dot(self.Whv, v) + np.dot(self.Who, o) + self.hb)))
		else:
			h_term = -1 * np.sum(np.log(1 + np.exp(h_partial + np.dot(self.Who, o))))
		return obias_term + vbias_term + h_term

	def __getstate__(self):
		d = {
			'nvis':		self.nvis,
			'nhid':		self.nhid,
			'nout':		self.nout,
			'vtype':	self.vtype,
			'htype':	self.htype,
			'otype':	self.otype,
			'Whv':		self.Whv.copy(),
			'Who':		self.Who.copy(),
			'vb':		self.vb.copy(),
			'hb':		self.hb.copy(),
			'ob':		self.ob.copy(),
			'dWhv':		self.dWhv.copy(),
			'dWho':		self.dWho.copy(),
			'dhb':		self.dhb.copy(),
			'dob':		self.dob.copy(),
			'dvb':		self.dvb.copy()}
		return d

	def __setstate__(self, d):
		self.nvis = 	d['nvis']
		self.nhid =		d['nhid']
		self.nout =		d['nout']
		self.vtype = 	d['vtype']
		self.htype = 	d['htype']
		self.otype = 	d['otype']
		self.Whv = 		d['Whv']
		self.Who = 		d['Who']
		self.vb = 		d['vb']
		self.hb =		d['hb']
		self.ob =		d['ob']
		self.dWhv =		d['dWhv']
		self.dWho =		d['dWho']
		self.dvb =		d['dvb']
		self.dhb =		d['dhb']
		self.dob =		d['dob']
		self.vact = unittypes[self.vtype]
		self.hact = unittypes[self.htype]
		self.oact = unittypes[self.otype]

