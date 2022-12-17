install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
	
format:	
	black *.py mylib/*py 

lint:
	# pylint --disable=R,C --ignore-patterns=test_.*?py *.py dblib
	pylint --disable=R,C *.py mylib/*.py

test:
	python -m pytest -vv --cov=mylib --cov=main test_*.py

all: install lint test format 