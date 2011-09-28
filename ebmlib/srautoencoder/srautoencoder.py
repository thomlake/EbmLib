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
#	srautoencoder.py
# description:
#	Simple Recursive Autoencoder Class
#---------------------------------------#

import numpy as np
from .. units import unittypes

class SimpleRecursiveAutoencoder(object):
	"""autoencoder class

	:param nin: number of input units
	:param nhid: number of hidden units
	:param htype: hidden unit type, see units.py for available types
	:param otype: output unit type, see units.py for available types

	:type nin: int
	:type nhid: int
	:type htype: string
	:type otype: string
	"""
	def __init__(self, nin, nhid, htype = 'tanh', otype = 'sigmoid'):
		self.nin = nin
		self.nhid = nhid
		self.i = np.zeros(nin)
		self.c = np.zeros(nhid)
		self.h = np.zeros(nhid)
		self.oi = np.zeros(nin)
		self.oc = np.zeros(nhid)

		self.hb = np.zeros(nhid)
		self.oib = np.zeros(nin)
		self.ocb = np.zeros(nhid)

		self.whi = np.random.normal(0., 0.2, (nhid, nin))
		self.whc = np.random.normal(0., 0.2, (nhid, nhid))
		self.woih = np.random.normal(0., 0.2, (nin, nhid))
		self.woch = np.random.normal(0, 0.2, (nhid, nhid))

		self.dwhi = np.zeros((nhid, nin))
		self.dwhc = np.zeros((nhid, nhid))
		self.dwoih = np.zeros((nin, nhid))
		self.dwoch = np.zeros((nhid, nhid))
		
		self.dhb = np.zeros(nhid)
		self.doib = np.zeros(nin)
		self.docb = np.zeros(nhid)

		self.htype = htype
		self.otype = otype

		self.hact = unittypes[htype]
		self.oact = unittypes[otype]

	def ff(self, x, c):
		"""get reconstruction of x

		:param x: input
		:param c: context
		:type x: numpy.array
		:type c: numpy.array
		:returns: reconstruction of x and c
		:rtype: (numpy.array, numpy.array)
		"""
		self.h = self.hact(np.dot(self.whi, x) + np.dot(self.whc, c) + self.hb)
		self.oi = self.oact(np.dot(self.woih, self.h) + self.oib)
		self.oc = self.hact(np.dot(self.woch, self.h) + self.ocb)
		return self.oi, self.oc

	def push(self, x):
		"""push an input x

		:param x: input
		:type x: numpy.array
		:returns: encoding of the current context given x
		:rtype: numpy.array
		"""
		self.h = self.hact(np.dot(self.whi, x) + np.dot(self.whc, self.h) + self.hb)
		return self.h

	def pop(self):
		"""pop an input vector and return the system to the previous context

		:returns: decoding of the most recent input
		:rtype: numpy.array
		"""
		self.oi = self.oact(np.dot(self.woih, self.h) + self.oib)
		self.h = self.hact(np.dot(self.woch, self.h) + self.ocb)
		return self.oi

	def reset(self):
		"""reset the netowrks stateful hidden units to 0
		
		:rtype: None
		"""
		self.h = np.zeros(self.nhid)

	def __getstate__(self):
		d = {
			'nin':		self.nin,
			'nhid':		self.nhid,
			'htype':	self.htype,
			'otype':	self.otype,
			'whi':		self.whi.copy(),
			'whc':		self.whc.copy(),
			'woih':		self.woih.copy(),
			'woch':		self.woch.copy(),
			'hb':		self.hb.copy(),
			'oib':		self.oib.copy(),
			'ocb':		self.ocb.copy(),
			'dwhi':		self.dwhi.copy(),
			'dwhc':		self.dwhc.copy(),
			'dwoih':	self.dwoih.copy(),
			'dwoch':	self.dwoch.copy(),
			'dhb':		self.dhb.copy(),
			'doib':		self.doib.copy(),
			'docb':		self.docb.copy(),
			'hstate':	self.h.copy()}
		return d

	def __setstate__(self, d):
		self.nin =		d['nin']
		self.nhid =		d['nhid']
		self.htype =	d['htype']
		self.otype =	d['otype']
		self.whi =		d['whi']
		self.whc =		d['whc']
		self.woih =		d['woih']
		self.woch =		d['woch']
		self.hb =		d['hb']
		self.oib =		d['oib']
		self.ocb =		d['ocb']
		self.dwhi =		d['dwhi']
		self.dwhc =		d['dwhc']
		self.dwoih =	d['dwoih']
		self.dwoch =	d['dwoch']
		self.dhb =		d['dhb']
		self.doib =		d['doib']
		self.docb =		d['docb']
		
		self.hact = unittypes[self.htype]
		self.oact = unittypes[self.otype]

		self.i = np.zeros(self.nin)
		self.c = np.zeros(self.nhid)
		self.h = d['hstate']
		self.oi = np.zeros(self.nin)
		self.oc = np.zeros(self.nhid)


