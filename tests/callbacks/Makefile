.PHONY: all test clean

all:
	python3 setup.py build_ext --inplace

test: all
	python3 run_cheese.py

clean:
	@rm -f cheese.c *.o *.so *~ core
	@rm -rf build
