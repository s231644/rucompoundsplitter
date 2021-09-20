from typing import List, Optional


class Vocab:
    def __init__(self, chars: Optional[List[str]] = None):
        self.i2w = ["<pad>", "<unk>", "<sos>", "<eos>"]

        self.i2w += chars or []
        self.w2i = {w: i for i, w in enumerate(self.i2w)}

    @property
    def pad_idx(self):
        return self.w2i["<pad>"]

    @property
    def unk_idx(self):
        return self.w2i["<unk>"]

    @property
    def sos_idx(self):
        return self.w2i["<sos>"]

    @property
    def eos_idx(self):
        return self.w2i["<eos>"]

    def tokenize(self, word: str):
        ids = [self.sos_idx]
        for c in word:
            ids.append(self.w2i.get(c, self.unk_idx))
        else:
            ids.append(self.eos_idx)
        return ids

    def detokenize(self, inds: list):
        res = []
        for i in inds:
            if i == self.unk_idx:
                res.append("?")
            elif i > 3:
                res.append(self.i2w[i])
            else:
                pass
        return "".join(res)

    @classmethod
    def from_list(cls, i2w):
        c = cls()
        c.i2w = i2w
        c.w2i = {w: i for i, w in enumerate(c.i2w)}
        return c


vocab = Vocab([
    ' ', '-',
    'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й',
    'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф',
    'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я'
])
