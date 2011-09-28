from distutils.core import setup

setup(
	name='EbmLib',
	version='0.1.0',
	author='tllake',
	author_email='thom.l.lake@gmail.com',
	packages=['ebmlib', 'ebmlib.rbm', 'ebmlib.srrbm', 'ebmlib.autoencoder', 'ebmlib.srautoencoder'],
	license='LICENSE.txt',
	description='Energy Based Models for Python.',
	long_description=open('README.rst').read(),
)

