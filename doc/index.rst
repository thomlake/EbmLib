.. EbmLib documentation master file, created by
   sphinx-quickstart on Thu Apr 28 18:31:53 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EbmLib Documentation
===================================
EbmLib is a Python library for working with energy based model, primarily created for my own research needs. EbmLib only depends on Numpy.

Models
-------------
* Restricted Boltzmann Machines (Hinton, Smolensky)
* AutoEncoders
* Recursive AutoEncoders **Coming**
* Recursive Restriced Boltzmann Machines (tllake)
* Oja's Linear Autencoder (tllake) **Coming**
* Recursive Oja's Linear Autencoder (tllake) **Coming**

Installation
============
installing (you may need sudo)::

    $ python setup.py install

Documentation
=============
You'll need `Sphinx <http://sphinx.pocoo.org/>`_ to build documentation.
Sphinx is available from PyPI or if you have easy install::
	
    $ easy_intall -U Sphinx

Once you've obtained Sphinx build html documentation::

	$ cd /path/to/EbmLib/doc/
	$ make html

pdf documentation is not currently working.

Modules
--------
.. toctree::
   :maxdepth: 2

   units.rst
   rbm.rst
   cdktrainer.rst
   autoencoder.rst
   backproptrainer.rst
   srrbm.rst
   recursive_cdktrainer.rst

About
-----

EbmLib is very beta. You may/might/will find bugs.

EbmLib is free software licensed under the `GNU General Public License <http://www.gnu.org/licenses/gpl.html>`_.

========    ======================================
author	    email       
========    ======================================
tllake      thomas dot l dot lake at wmich dot edu 
========    ======================================

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


