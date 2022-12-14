FROM python:3.10-slim-buster
LABEL MAINTAINER=sc765@duke.edu

WORKDIR /app/
# COPY ./interface.py ./
COPY ./app_image.py ./

COPY ./images ./images

COPY ./requirements_docker.txt ./requirements.txt
COPY ./Makefile ./

RUN apt-get update && apt-get install make
RUN make install 

RUN python -c "from transformers import pipeline; pipeline('text-classification',model='Shunian/mbti-classification-roberta-base', top_k=1)" && \
    python -c "import transformers; transformers.utils.move_cache()"

EXPOSE 5000

CMD ["python", "app_image.py"]