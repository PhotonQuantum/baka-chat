import datetime
import json
import pickle
import time

import oss2

from ..common.dataset import Corpus
from ..common.oss import BaseOSS


class TrainerOSS(BaseOSS):
    def get_chat_log(self, group_id: int) -> list:
        """ Returns a list contains all chat logs in specific chat group. """
        whole_history = []

        # Multiple files are read and all logs are concatenated.
        for file in oss2.ObjectIterator(self.bucket, prefix="dataset/"):
            raw_log = pickle.loads(self.bucket.get_object(file.key).read())
            dataset = raw_log.get(group_id, None)
            if dataset:
                whole_history.extend(dataset)

        return whole_history

    def upload_model(self, corpus: Corpus, meta_data: dict, model: bytes, debug: bool = False):
        """ Upload the corpus, model, metadata and training log. """
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        ts = int(time.time())

        prefix = "debug_model_v2" if debug else "model_v2"
        self.bucket.put_object(f"{prefix}/{now}_{ts}/corpus.json", json.dumps(corpus.dumps()))
        self.bucket.put_object(f"{prefix}/{now}_{ts}/meta.json", json.dumps(meta_data))
        self.bucket.put_object(f"{prefix}/{now}_{ts}/net.pth", model)

    def clean_model(self, keep_models: int = 5):
        """ Clean old models. Only keep_models recent ones are kept. """
        models = [entry.key for entry in oss2.ObjectIterator(self.bucket, prefix="model_v2/")]
        if not models:
            return

        models = [entry.split("/")[1] for entry in models]  # get model names
        models.sort(key=lambda x: x.split("_")[1])  # sort them by date
        delete_list = []
        for model in models[:-keep_models]:
            path = "/".join(["model_v2", model])
            for file in oss2.ObjectIterator(self.bucket, prefix=path):
                delete_list.append(file.key)  # all files in the folder should be deleted before removing it
            delete_list.append(path)  # now we can safely delete the folder

        if delete_list:
            self.bucket.batch_delete_objects(delete_list)  # we perform the bulk delete operation
