import re
from typing import Optional

import ftfy

# noinspection RegExpRedundantEscape
sub_list = [
    (re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'), ""),
    (re.compile(r'@.*?;'), ";"),
    (re.compile(r'@.*? '), ""),
    (re.compile(r'\[(图片|表情|短视频)\]'), ""),
    (re.compile(r'�'), "")
]


def clean_text(text: str, nickname_set: Optional[set] = None) -> str:
    """ Remove unnecessary symbols and normalize sentence. """
    nickname_set = nickname_set if nickname_set else set()

    for pattern, sub in sub_list:
        text = re.sub(pattern, sub, text)  # clean the text with predefined regex patterns
    for name in nickname_set:
        text = text.replace(f"@{name}", "")  # now we remove all nickname.

    return ftfy.fix_text(text)  # normalize the text
