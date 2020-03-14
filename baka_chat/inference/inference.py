from typing import List

import torch

from ..common.dataset import Corpus
from ..common.model import RNNModule
from ..common.utils import clean_text


class InferenceSession(RNNModule):
    def __init__(self, corpus: Corpus, hparams: dict, n: int = 2, temp: float = 1.5):
        super().__init__(corpus, hparams)
        self.batch_size = 1  # for online inference
        self.n = n
        self.temp = temp
        self.eos = self.corpus.table_vocab_to_int[";"]

    def _inference(self, sequence: List[int]) -> List[List[int]]:
        """
        Input: A sequence of word vectors, split with eos.
        Output: Multiple sequences of word vectors. Each sequence represents a complete sentence.
        """
        self.eval()
        with torch.no_grad():
            state_h, state_c = self.zero_state()
            for w in sequence[:-1]:
                ix = torch.tensor([[w]]).cpu()
                output, (state_h, state_c) = self(ix, (state_h, state_c))

            choice = sequence[-1]
            result = []
            temp_seq = []
            counter = 0
            while counter < self.n:
                ix = torch.tensor([[choice]]).cpu()
                output, (state_h, state_c) = self(ix, (state_h, state_c))

                probability = output[0][0].div(self.temp).exp()
                choice = int(torch.multinomial(probability, 1)[0])

                if choice == self.eos:
                    result.append(temp_seq.copy())
                    temp_seq.clear()
                    counter += 1
                else:
                    temp_seq.append(choice)

        return result

    def inference_seq(self, sequence: List[str]) -> List[List[str]]:
        """
        Input: A sequence of words, split with eos.
        Output: Multiple sequences of words. Each sequence represents a complete sentence.
        """
        input_vector = self.corpus.word_to_int(sequence)
        output_vector = self._inference(input_vector)
        output = [self.corpus.int_to_word(sentence) for sentence in output_vector]
        return output

    def inference_sentence(self, sentence: str) -> str:
        """
        Input: A sentence/sentences split with eos.
        Output: Sentences split with eos.
        """
        sequence = self._match_vocab(clean_text(sentence, self.corpus.nickname_set))
        sentences = self.inference_seq(sequence)
        return ";".join(["".join(sentence) for sentence in sentences])

    def _match_vocab(self, sentence: str) -> List[str]:
        vocab = list(self.corpus.table_vocab_to_int.keys())
        vocab.sort(key=len, reverse=True)
        output = []
        while sentence:
            flag = False
            for entry in vocab:
                try:
                    if sentence[:len(entry)] == entry:
                        output.append(entry)
                        sentence = sentence[len(entry):]
                        flag = True
                        break
                except IndexError:
                    continue
            if not flag:
                sentence = sentence[1:]
        return output
