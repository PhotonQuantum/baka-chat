from dateutil import parser
import bisect
import pickle
import re
from datetime import datetime
from utils import cut_text
from dataset import TextLib
import unicodedata

arabic_table = (
    (0x0600, 0x061e),
    (0x0620, 0x06ff),
    (0x0750, 0x077f),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
    (0x10E60, 0x10E7F),
    (0x1EC70, 0x1ECBF),
    (0x1ED00, 0x1ED4F),
    (0x1EE00, 0x1EEFF)
)
arabic_table_flatten = []
for x, y in arabic_table:
    arabic_table_flatten.extend([x, y+1])

meta_pattern_legacy = re.compile(r"(.*)\((\d{6,})\)\s*(\d{2}:\d{2}:\d{2})")
meta_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}) (.*)\((\d*)\)')
reply_pattern = re.compile(r'" .* \d{2}:\d{2}:\d{2}[\s\S]*? "\n')

def is_arabic(char):
    return bool(bisect.bisect(arabic_table_flatten, ord(char)) % 2)

def legacy_loader(f, date):
    text = ""
    result = []
    nickname, uid, time_ = "", 0, ""
    arabic = False
    with open(f, encoding="utf-8") as f:
        whole = re.sub(reply_pattern, "", f.read())
    for line in whole.split("\n"):
        match = re.search(meta_pattern_legacy, line)
        if match:
            meta = match.groups()
            if text:
                if uid != "1431246122":
                    result.append({
                        "nickname": nickname,
                        "id": uid,
                        "datetime": parser.parse(time_, default=date),
                        "text": text.rstrip().rstrip("("),
                        "meta": {
                            "arabic": arabic
                        }
                    })
                text = ""
                arabic = False
            nickname, uid, time_ = meta
            # dt, uid, nickname = meta
        else:
            for char in line:
                if is_arabic(char):
                    arabic = True
                    break
            text += line
    return result

def loader(f):
    text = ""
    result = []
    nickname, uid, dt = "", 0, ""
    arabic = False
    with open(f, encoding="utf-8") as f:
        whole = re.sub(reply_pattern, "", f.read())
    for line in whole.split("\n"):
        match = re.search(meta_pattern, line)
        if match:
            meta = match.groups()
            if text:
                if uid != "1431246122":
                    result.append({
                        "nickname": nickname,
                        "id": uid,
                        "datetime": parser.parse(dt),
                        "text": text.rstrip().rstrip("("),
                        "meta": {
                            "arabic": arabic
                        }
                    })
                text = ""
                arabic = False
            # nickname, uid, time_ = meta
            dt, nickname, uid = meta
        else:
            for char in line:
                if is_arabic(char):
                    arabic = True
                    break
            text += line
    return result

chat_history = []
chat_history.extend(loader("base.txt"))
chat_history.extend(legacy_loader("20200303.txt", datetime(2020, 3, 3, 0, 0, 0)))
chat_history.extend(legacy_loader("20200304.txt", datetime(2020, 3, 4, 0, 0, 0)))
chat_history.extend(loader("20200305_new.txt"))

with open("output.pickle", mode="wb") as f:
    pickle.dump(chat_history, f)

text_lib = TextLib(cut_text(list(filter(lambda x: not x["meta"]["arabic"], chat_history))))
with open("text_lib.vocab", mode="wb") as f:
    pickle.dump(text_lib, f)
