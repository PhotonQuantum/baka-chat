import json
import sys

import torch

from .inference import InferenceSession
from ..common.dataset import Corpus


def main():
    hparams = json.load(sys.argv[1])
    corpus = Corpus.loads(json.load(sys.argv[2]))
    session = InferenceSession(corpus, hparams)
    session.load_state_dict(torch.load(sys.argv[3], map_location=torch.device("cpu")))
    session.n = 5
    for _ in range(5):
        print(session.inference_sentence(sys.argv[4]))


if __name__ == "__main__":
    main()
