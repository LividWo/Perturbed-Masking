import os
from itertools import count


class Token:
    pass


class UToken(Token):
    def __init__(self, tid, form, lemma, upos, xpos, feats,
                 head, deprel, deps, misc):
        """
        Args:
          tid: Word index, starting at 1; may be a range for multi-word tokens;
            may be a decimal number for empty nodes.
          form: word form or punctuation symbol.
          lemma: lemma or stem of word form
          upos: universal part-of-speech tag
          xpos: language specific part-of-speech tag
          feats: morphological features
          head: head of current word (an ID or 0)
          deprel: universal dependency relation to the HEAD (root iff HEAD = 0)
          deps: enhanced dependency graph in the form of a list of head-deprel pairs
          misc: any other annotation
        """
        self.str_id = tid  # Use this for printing the conll
        self.id = int(float(tid))  # Use this for training TODO: what is this 10.1 business?
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = int(head)
        self.deprel = deprel
        self.deps = deps
        self.misc = misc

    def __str__(self):
        return '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
            self.str_id, self.form, self.lemma, self.upos, self.xpos, self.feats,
            self.head, self.deprel, self.deps, self.misc)

    @property
    def pos(self):
        return self.upos


def read_conllu(f):

    tokens = []

    for line in f:
        line = line.strip()

        if not line:
            yield tokens
            tokens = []
            continue

        if line[0] == "#":
            continue

        parts = line.split()
        assert len(parts) == 10, "invalid conllu line"
        tokens.append(UToken(*parts))

    # possible last sentence without newline after
    if len(tokens) > 0:
        yield tokens


def print_example(ex):

    if "head" in ex.__dict__.keys():
        r = ["%2d %12s %5s -> %2d (%s)" % (i, f, p, h, d) for i, f, p, h, d in zip(
            count(start=1), ex.form, ex.pos, ex.head, ex.deprel)]
    else:
        r = ["%2d %12s %5s -> ? ?" % (i, f, p) for i, f, p in zip(
            count(start=1), ex.form, ex.pos)]
    print("\n".join(r))
    print()


class XToken(Token):
    """Conll-X Token Representation"""

    def __init__(self, tid, form, lemma, cpos, pos, feats,
               head, deprel, phead, pdelrel):
        self.id = int(tid)
        self.form = form
        self.lemma = lemma
        self.cpos = cpos
        self.pos = pos
        self.feats = feats
        self.head = int(head)
        self.deprel = deprel
        self.phead = phead
        self.pdeprel = pdelrel

    def __str__(self):
        return '%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s' % (
            self.id, self.form, self.lemma, self.cpos, self.pos, self.feats,
            self.head, self.deprel, self.phead, self.pdeprel)

    def __repr__(self):
        return self.__str__()



if __name__ == '__main__':
  pass