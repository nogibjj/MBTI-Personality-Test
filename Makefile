install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt\
		pip install torch --extra-index-url https://download.pytorch.org/whl/cu116\
		pip cache purge

format:
	black *.py

lint:
	pylint --output-format=colorized --disable=R,C,W1203,W1202,W1514 *.py
test:
	python test.py

deploy:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 709249667281.dkr.ecr.us-east-1.amazonaws.com
	docker build -t ids706 .
	docker tag ids706:latest 709249667281.dkr.ecr.us-east-1.amazonaws.com/ids706:latest
	docker push 709249667281.dkr.ecr.us-east-1.amazonaws.com/ids706:latest
	
all: install format lint test