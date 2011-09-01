EbmLib Documentation
===================================
EbmLib is a Python library for working with energy based model, primarily created for my own research needs.

Models
-------------
* Restricted Boltzmann Machines (Hinton, Smolensky)
* Recursive Restriced Boltzmann Machines (tllake)
* AutoEncoders
* Recursive AutoEncoders **TODO**
* Oja's Linear Autencoder **TODO**
* Recursive Oja's Linear Autencoder (tllake) **TODO**

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

About
-----
EbmLib is very beta. You may/might/will find bugs.

EbmLib is free software licensed under the `GNU General Public License <http://www.gnu.org/licenses/gpl.html>`_.

========    ======================================
author	    email       
========    ======================================
tllake      thomas dot l dot lake at wmich dot edu 
========    ======================================

