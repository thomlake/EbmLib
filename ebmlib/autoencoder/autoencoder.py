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
#	autoencoder.py
# description:
#	Autoencoder Class
#---------------------------------------#

import numpy as np
from .. units import unittypes

class AutoEncoder(object):
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
		self.h = np.zeros(nhid)
		self.o = np.zeros(nin)

		self.hb = np.zeros(nhid)
		self.ob = np.zeros(nin)

		self.whi = np.random.normal(0., 0.2, (nhid, nin))
		self.woh = np.random.normal(0., 0.2, (nin, nhid))

		self.dwhi = np.zeros((nhid, nin))
		self.dwoh = np.zeros((nin, nhid))
		
		self.dhb = np.zeros(nhid)
		self.dob = np.zeros(nin)

		self.htype = htype
		self.otype = otype

		self.hact = unittypes[htype]
		self.oact = unittypes[otype]

	def ff(self, x):
		"""get reconstruction of x

		:param x: input
		:type x: numpy.array
		:returns: reconstruction of x
		:rtype: numpy.array
		"""
		self.h = self.hact(np.dot(self.whi, x) + self.hb)
		self.o = self.oact(np.dot(self.woh, self.h) + self.ob)
		return self.o

	def encode(self, x):
		"""get encoding of x

		:param x: input
		:type x: numpy.array
		:returns: encoding of x
		:rtype: numpy.array
		"""
		self.h = self.hact(np.dot(self.whi, x) + self.hb)
		return self.h

	def decode(self, h):
		"""get decoding of h

		:param h: hidden state to decode
		:type h: numpy.array
		:returns: decoding of h
		:rtype: numpy.array
		"""
		self.o = self.oact(np.dot(self.woh, h) + self.ob)

