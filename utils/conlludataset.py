import io
import os

from utils import UToken

ROOT_TOKEN = '<root>'
ROOT_TAG = 'ROOT'
ROOT_LABEL = '-root-'


def empty_conllu_example_dict():
    ex = {
        'id':      [],
        'form':    [],
        'lemma':   [],
        'pos':     [],
        'upos':    [],
        'feats':   [],
        'head':    [],
        'deprel':  [],
        'deps':    [],
        'misc':    [],
    }
    return ex


def root():
    ex = {
        'id': [0],
        'form': [ROOT_TOKEN],
        'lemma': [ROOT_TOKEN],
        'pos': [ROOT_TAG],
        'upos': [ROOT_TAG],
        'feats': ['_'],
        'head': [0],
        'deprel': [ROOT_LABEL],
        'deps': ['_'],
        'misc': ['_'],
    }
    return ex


def conllu_reader(f):
    ex = root()

    for line in f:
        line = line.strip()

        if not line:
            yield ex
            ex = root()
            continue

        # comments
        if line[0] == "#":
            continue

        parts = line.split()
        assert len(parts) == 10, "invalid conllx line: %s" % line

        _id, _form, _lemma, _upos, _xpos, _feats, _head, _deprel, _deps, _misc = parts

        ex['id'].append(_id)
        ex['form'].append(_form)
        ex['lemma'].append(_lemma)
        ex['upos'].append(_upos)
        ex['pos'].append(_xpos)
        ex['feats'].append(_feats)

        # TODO: kan dit? (0 is root)
        if _head == "_":
            _head = 0

        ex['head'].append(_head)
        ex['deprel'].append(_deprel)
        ex['deps'].append(_deps)
        ex['misc'].append(_misc)

    # possible last sentence without newline after
    if len(ex['form']) > 0:
        yield ex


class ConllUDataset:
    """Defines a CONLL-U Dataset. """

    def __init__(self, path):
        print('dataset used:', os.path.expanduser(path))
        """Create a ConllUDataset given a path and field list.
        Arguments:
            path (str): Path to the data file.
            fields (dict[str: tuple(str, Field)]):
                The keys should be a subset of the columns, and the
                values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
        """

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            self.examples = [d for d in conllu_reader(f)]

        # for d in self.examples:
        #     token = []
        #     for parts in zip(*d.values()):
        #         token.append(UToken(*parts))
            # self.tokens.append(token)

        # this line is a pythonic version of the above fuction
        # d is a whole sentence
        # d.values gets all values of the dict
        # 单星号（*）：*agrs 将所以参数以元组(tuple)的形式导入：
        # parts represents one line in conllu file
        self.tokens = [
            [UToken(*parts) for parts in zip(*d.values())]
            for d in self.examples]


# c = ConllUDataset('./data/EWT/en_ewt-ud-test.conllu')
# print('done')