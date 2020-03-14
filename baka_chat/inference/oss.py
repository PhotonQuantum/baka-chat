import json
from datetime import datetime
from os import path
from typing import Tuple

import oss2

from ..common.oss import BaseOSS


class InferenceOSS(BaseOSS):
    PREFIX = "model_v2/"

    def list_models(self):
        # model_v2/2020-03-12_1584028278
        models = [entry.key for entry in oss2.ObjectIterator(self.bucket, prefix=self.PREFIX)]
        if not models:
            raise RuntimeError("No model available.")

        models = [entry.split("/")[1] for entry in models]  # 2020-03-12_1584028278
        models.sort(key=lambda x: x.split("_")[1])

        return models

    def download_model(self, key: str) -> Tuple[dict, dict, bytes]:
        meta_path = path.join(self.PREFIX, key, "meta.json")
        corpus_path = path.join(self.PREFIX, key, "corpus.json")
        weight_path = path.join(self.PREFIX, key, "net.pth")

        meta_dict = json.load(self.bucket.get_object(meta_path))
        corpus_dict = json.load(self.bucket.get_object(corpus_path))
        net_weight = self.bucket.get_object(weight_path).read()

        return meta_dict, corpus_dict, net_weight

    @staticmethod
    def key_to_version(key: str) -> str:
        # key: 2020-03-12_1584028278
        return datetime.fromtimestamp(int(key.split("_")[1])).strftime('%Y-%m-%d_%H:%M:%S')
