import os

from comet_ml import Experiment
from loguru import logger

from .config import hparams
from .oss import TrainerOSS
from .trainer import Trainer
from .utils import get_nickname_set, preprocess_log
from ..common.dataset import Corpus

production = os.environ.get("PRODUCTION_MODE") == "yes"
oss = TrainerOSS(os.environ["ALIYUN_ACCESSKEY_ID"], os.environ["ALIYUN_ACCESSKEY_SECRET"], "baka-bot-data", production)

experiment = Experiment(os.environ["COMET_KEY"], project_name="baka-bot")


def main():
    logger.info("Fetching dataset.")
    chat_log = oss.get_chat_log(883143987)
    name_set = get_nickname_set(chat_log)
    pre_log = preprocess_log(chat_log, name_set)
    corpus = Corpus(pre_log, name_set)

    logger.info("Cleaning model.")
    oss.clean_model()

    logger.info("Training.")
    net = Trainer(corpus, hparams)
    best_model = net.fit(experiment)

    logger.info("Uploading.")

    # Save state_dict to memory and read it.
    model = net.dumps(best_model["state_dict"])

    meta = hparams.copy()
    meta.update({"loss": best_model["loss"], "epoch": best_model["epoch"]})
    oss.upload_model(corpus, meta, model, debug=not production)

    logger.info("Finished.")


if __name__ == "__main__":
    main()
