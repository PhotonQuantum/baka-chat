# Baka Chat

A toy chat bot which uses a simple LSTM model to generate replies to inputs.

This project uses Aliyun OSS to store datasets and models.

> WIP
>
> This readme is under construction, and I'm writing a blog post on this toy project.

## Installation

There are two dockers in this projects, all requiring the `baka_chat` python package.

- *Train* - Pulls dataset from Object Storage, trains the model, uploads training log to comet.ml,
and pushes the corpus and model weights bak to Object Storage.
- *Inference* - Downloads pretrained model and corpus, and brings up an HTTP server, providing
inference service.

## Built With

- [PyTorch](https://pytorch.org/) - An open source machine learning framework that accelerates the path from research prototyping to production deployment.
- [jieba](https://github.com/fxsjy/jieba) - Chinese text segmentation: built to be the best Python Chinese word segmentation module.
- [ftfy](https://github.com/LuminosoInsight/python-ftfy) - Fixes mojibake and other glitches in Unicode text, after the fact.
- [Starlette](https://www.starlette.io/) - A lightweight ASGI framework/toolkit, which is ideal for building high performance asyncio services.
- [Uvicorn](https://www.uvicorn.org/) - A lightning-fast ASGI server, built on uvloop and httptools.
- [Loguru](https://github.com/Delgan/loguru) - A library which aims to bring enjoyable logging in Python.
- [Docker](https://www.docker.com/) - Securely build and share any application, anywhere.
- [Aliyun OSS Python SDK](https://github.com/aliyun/aliyun-oss-python-sdk) - Alibaba Cloud Object Storage Python SDK 2.x.

See [Dockerfile](Dockerfile) and [Dockerfile_train](Dockerfile_train) for details.

## License

This project is licensed under GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.