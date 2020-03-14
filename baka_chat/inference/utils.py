from io import BytesIO

import torch

from .inference import InferenceSession
from .oss import InferenceOSS
from ..common.dataset import Corpus


def load_model(model_key: str, oss: InferenceOSS):
    meta, corpus_dict, net_weight = oss.download_model(model_key)

    corpus = Corpus.loads(corpus_dict)
    net_weight_file = BytesIO(net_weight)
    net_weight_file.seek(0)

    session = InferenceSession(corpus, meta)
    net_state_dict = torch.load(net_weight_file, map_location=torch.device("cpu"))
    session.load_state_dict(net_state_dict)

    return session
