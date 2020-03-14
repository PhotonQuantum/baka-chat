FROM python:3.7.6-slim

MAINTAINER LightQuantum

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install torch==1.4.0+cpu numpy -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install ftfy

RUN pip install oss2

RUN pip install starlette uvicorn

COPY ./baka_chat ./baka_chat

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "80", "baka_chat.inference.app:app"]
