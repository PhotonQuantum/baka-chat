import re
from typing import List, Set

import jieba

from ..common.utils import clean_text


def get_nickname_set(chat_log: list) -> Set[str]:
    """ Get all nicknames in the chat log. """
    name_set = set()
    for entry in chat_log:
        name_set.add(entry["nickname"])

    return name_set


def preprocess_log(chat_log: list, nickname_set: set) -> List[str]:
    """ Clean the chat log and do text segmentation. """
    # Prepare a nickname set, and we will remove all username in this set.
    chat_log = filter(lambda x: x["id"] != "1283637358", chat_log)  # removes all bot generated sentences.
    chat_log = list(filter(lambda x: not x["meta"]["arabic"], chat_log))  # removes all arabic sentences.

    # Concatenate texts.
    joint_text = ""
    for entry in chat_log:
        cleaned_entry = clean_text(entry["text"], nickname_set)
        joint_text += cleaned_entry + ";"  # add stop symbol

    # Clean multiple spaces and stop symbols.
    for pattern, sub in [(r"\s\s+", " "), (r";+", ";")]:
        joint_text = re.sub(pattern, sub, joint_text)

    # Do segmentation
    rtn = jieba.lcut(joint_text)

    return rtn
