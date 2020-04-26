import collections


def undirected_standard(gold):
    undirected = [(head, dependent) for (dependent, head) in gold]
    return gold+undirected


def ned_standard(gold):
    undirected = [(head, dependent) for (dependent, head) in gold]
    tree = collections.defaultdict(lambda: -1)
    for (dependent, head) in gold:
        tree[dependent] = head
    ned = []
    for (dependent, head) in gold:
        grandparent = tree[head]
        if grandparent != -1:
            ned.append((dependent, grandparent))

    return gold + undirected + ned


def tag_evaluation(trees, results, deprels):
    n_correct, n_incorrect = collections.Counter(), collections.Counter()
    for tree, result, deprel in zip(trees, results, deprels):
        line, tokenized_text, matrix_as_list = result
        directed_gold_edges = [(x.id, x.head) for x in line]
        for i, (p, g, r) in enumerate(zip(tree, directed_gold_edges, deprel)):
            if i == 0:
                continue
            is_correct = (p == g)
            if is_correct:
                n_correct[r] += 1
                n_correct['all'] += 1
            else:
                n_incorrect[r] += 1
                n_incorrect['all'] += 1
    by_tag_accuracy = dict()
    for k in n_correct.keys():
        by_tag_accuracy[k] = n_correct[k]/float(n_correct[k] + n_incorrect[k])
    for k,v in sorted(by_tag_accuracy.items(), key=lambda x:x[1], reverse=True):
        if n_correct[k] + n_incorrect[k] < 20:
            continue
        print('{}\t\t\t{:.2f}\t{}\t{}'.format(k, v, 100 * n_correct[k] / float(n_correct[k] + n_incorrect[k]),
                                        n_correct[k] + n_incorrect[k], n_correct[k] + n_incorrect[k]))
    return by_tag_accuracy


def new_evaluation(trees, results):
    uas_count, total_relations = 0., 0.
    uuas_count = 0.
    ned_count = 0.
    # uls_count = 0.
    for tree, result in zip(trees, results):
        line, tokenized_text, matrix_as_list = result
        directed_gold_edges = [(x.id, x.head) for x in line][1:]
        # uls_count += tree_score(tree[1:], directed_gold_edges)
        undirected_gold_edges = undirected_standard(directed_gold_edges)
        ned_gold_edges = ned_standard(directed_gold_edges)

        identical = set(directed_gold_edges) & set(tree)
        undirected_identical = set(undirected_gold_edges) & set(tree)
        ned_identical = set(ned_gold_edges) & set(tree)

        total_relations += len(directed_gold_edges)
        uas_count += len(identical)
        uuas_count += len(undirected_identical)
        ned_count += len(ned_identical)
    uas = uas_count / total_relations
    uuas = uuas_count / total_relations
    ned = ned_count / total_relations
    # uls = uls_count / total_relations
    # print("UAS, UUAS, NED, ULS:", uas, uuas, ned, uls)
    print("UAS, UUAS, NED, ULS:", uas, uuas, ned)
    print("correct and total arcs", uas_count, total_relations)
    return uas, uuas, ned


def distance_tag_analysis(trees, results, deprels):
    n_correct, n_incorrect = collections.Counter(), collections.Counter()
    for tree, result, deprel in zip(trees, results, deprels):
        line, tokenized_text, matrix_as_list = result
        directed_gold_edges = [(x.id, x.head) for x in line]
        for i, (p, g, r) in enumerate(zip(tree, directed_gold_edges, deprel)):
            if i == 0:
                continue
            is_correct = (p == g)
            dist = abs(g[0]-g[1])
            if dist != 1:
                continue
            if is_correct:
                n_correct[r] += 1
            else:
                n_incorrect[r] += 1
    by_tag_accuracy = dict()
    for k in n_incorrect.keys():
        by_tag_accuracy[k] = n_correct[k] / float(n_correct[k] + n_incorrect[k])
    for k, v in sorted(by_tag_accuracy.items(), key=lambda x: x[1], reverse=True):
        if n_correct[k] + n_incorrect[k] < 20:
            continue
        print('{}\t\t{:.2f}\t{}'.format(k, 100 * n_correct[k] / float(n_correct[k] + n_incorrect[k]),
                                          n_correct[k] + n_incorrect[k]))


def distance_analysis(trees, results):
    n_correct, n_incorrect = collections.Counter(), collections.Counter()
    avg_pred_depth, avg_gold_depth = 0., 0.
    for tree, result in zip(trees, results):
        line, tokenized_text, matrix_as_list = result
        directed_gold_edges = [(x.id, x.head) for x in line]
        # directed_gold_edges = [(x.head, x.id) for x in line]
        for i, (p, g) in enumerate(zip(tree, directed_gold_edges)):
            if i == 0:
                continue
            is_correct = (p == g)
            dist = abs(g[0]-g[1])
            avg_gold_depth += dist
            avg_pred_depth += abs(p[0]-p[1])
            if is_correct:
                n_correct[dist] += 1
            else:
                n_incorrect[dist] += 1
    for k in sorted(n_correct.keys()):
        # if n_correct[k] + n_incorrect[k] < 20:
        #     continue
        print('{}\t{:.2f}\t{:.2f}'.format(k, 100 * n_correct[k] / float(n_correct[k] + n_incorrect[k]),
                                      100 * (n_correct[k] + n_incorrect[k]) / 21183.))
    print('average distance pred vs gold:', avg_pred_depth/21183., avg_gold_depth/21183.)


def no_punc_evaluation(trees, results, deprels):
    uas_count, total_relations = 0., 0.
    uuas_count = 0.
    for tree, result, deprel in zip(trees, results, deprels):
        deprel = deprel[1:]
        line, tokenized_text, matrix_as_list = result
        directed_gold_edges = []
        for idx, x in enumerate(line[1:]):
            if deprel[idx] == 'punct':
                continue
            directed_gold_edges.append((x.id, x.head))

        identical = set(directed_gold_edges) & set(tree)
        undirected_tree = [(head, dependent) for (dependent, head) in tree]
        undirected_tree += tree
        undirected_identical = set(directed_gold_edges) & set(undirected_tree)
        total_relations += len(directed_gold_edges)
        uas_count += len(identical)
        uuas_count += len(undirected_identical)
    uas = uas_count / total_relations
    uuas = uuas_count / total_relations
    print("uas and uuas:", uas, uuas)
    print("correct and total arcs", uas_count, total_relations)