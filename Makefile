# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace cython

clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

trailing-spaces:
	find spatch -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

cython:
	find spatch -name "*.pyx" -exec $(CYTHON) {} \;

doc: inplace
	$(MAKE) -C doc html

doc-noplot: inplace
	$(MAKE) -C doc html-noplot

code-analysis:
	flake8 spatch | grep -v __init__ | grep -v external
	pylint -E -i y spatch/ -d E1103,E0611,E1101
