import os
import time

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .oss import InferenceOSS
from .utils import load_model

production = os.environ.get("PRODUCTION_MODE") == "yes"
oss = InferenceOSS(os.environ["ALIYUN_ACCESSKEY_ID"], os.environ["ALIYUN_ACCESSKEY_SECRET"], "baka-bot-data",
                   production)

current_version = oss.list_models()[-1]

session = load_model(current_version, oss)


async def infer(request: Request):
    temp = float(request.query_params.get("temp", 1))
    n = int(request.query_params.get("n", 5))
    sentence = request.query_params.get("sentence", None)
    seq = request.query_params.get("seq", None)

    session.temp = temp
    session.n = n

    t_begin = time.time()

    if sentence:
        result = session.inference_sentence(sentence)
    elif seq:
        result = session.inference_seq(seq)
    else:
        return JSONResponse({"error": "Missing sentence or seq field."})

    t_end = time.time()

    return JSONResponse({"response": result, "temperature": temp, "n": n, "time": int((t_end - t_begin) * 1000)})


# noinspection PyUnusedLocal
async def stat(request: Request):
    # current_version: 2020-03-12_1584028278
    version_str = oss.key_to_version(current_version)
    return JSONResponse({"version": version_str, "net_meta": session.hparams})


async def update(request: Request):
    global session
    global current_version
    is_force = request.query_params.get("force", False) == "yes"

    latest_version = oss.list_models()[-1]

    # current_version: 2020-03-12_1584028278
    current_version_ts = int(current_version.split("_")[1])
    latest_version_ts = int(latest_version.split("_")[1])

    if is_force or current_version_ts < latest_version_ts:
        session = load_model(latest_version, oss)
        old_version = current_version
        current_version = latest_version
        return JSONResponse({"updated": True,
                             "version": oss.key_to_version(current_version),
                             "old": oss.key_to_version(old_version)})
    return JSONResponse({"updated": False, "version": oss.key_to_version(current_version)})


routes = [
    Route("/infer", infer),
    Route("/stat", stat),
    Route("/update", update)
]

app = Starlette(routes=routes)
