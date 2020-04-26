from dependency.evaluation import new_evaluation, tag_evaluation, distance_analysis, distance_tag_analysis, no_punc_evaluation
import argparse
from tqdm import tqdm
import numpy as np
import pickle
from dependency import DependencyDecoder, chuliu_edmonds
import unicodedata


def find_root(parse):
    # root node's head also == 0, so have to be removed
    for token in parse[1:]:
        if token.head == 0:
            return token.id
    return False


def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def match_tokenized_to_untokenized(subwords, sentence):
    token_subwords = np.zeros(len(sentence))
    sentence = [_run_strip_accents(x) for x in sentence]
    token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
    for i, subword in enumerate(subwords):
        if subword in ["[CLS]", "[SEP]"]: continue

        while current_token_normalized is None:
            current_token_normalized = sentence[current_token].lower()

        if subword.startswith("[UNK]"):
            unk_length = int(subword[6:])
            subwords[i] = subword[:5]
            subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
        else:
            subwords_str += subword[2:] if subword.startswith("##") else subword
        if not current_token_normalized.startswith(subwords_str):
            return False

        token_ids[i] = current_token
        token_subwords[current_token] += 1
        if current_token_normalized == subwords_str:
            subwords_str = ""
            current_token += 1
            current_token_normalized = None

    assert current_token_normalized is None
    while current_token < len(sentence):
        assert not sentence[current_token]
        current_token += 1
    assert current_token == len(sentence)

    return token_ids


def decoding(args):
    trees = []
    deprels = []
    with open(args.matrix, 'rb') as f:
        results = pickle.load(f)
    new_results = []
    decoder = DependencyDecoder()
    root_found = 0

    for (line, tokenized_text, matrix_as_list) in tqdm(results):
        orginal_line = line
        sentence = [x.form for x in line][1:]
        deprels.append([x.deprel for x in line])
        root = find_root(line)

        mapping = match_tokenized_to_untokenized(tokenized_text, sentence)

        init_matrix = matrix_as_list

        # merge subwords in one row
        merge_column_matrix = []
        for i, line in enumerate(init_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    if args.subword == 'max':
                        new_row.append(max(buf))
                    elif args.subword == 'avg':
                        new_row.append((sum(buf) / len(buf)))
                    elif args.subword == 'first':
                        new_row.append(buf[0])
                    buf = []
            merge_column_matrix.append(new_row)

        # merge subwords in multi rows
        # transpose the matrix so we can work with row instead of multiple rows
        merge_column_matrix = np.array(merge_column_matrix).transpose()
        merge_column_matrix = merge_column_matrix.tolist()
        final_matrix = []
        for i, line in enumerate(merge_column_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    if args.subword == 'max':
                        new_row.append(max(buf))
                    elif args.subword == 'avg':
                        new_row.append((sum(buf) / len(buf)))
                    elif args.subword == 'first':
                        new_row.append(buf[0])
                    buf = []
            final_matrix.append(new_row)

        # transpose to the original matrix
        final_matrix = np.array(final_matrix).transpose()

        if final_matrix.shape[0] == 0:
            print('find empty matrix:',sentence)
            continue
        assert final_matrix.shape[0] == final_matrix.shape[1]

        new_results.append((orginal_line, tokenized_text, matrix_as_list))

        if args.favor_local:
            weight = 0.001
            dist_mat = np.zeros(final_matrix.shape)
            n = final_matrix.shape[0]
            for i, row in enumerate(dist_mat):
                for j, cell in enumerate(row):
                    dist_mat[i, j] = abs(j-i)
                    # dist_mat[i, j] = abs(j-i)/n

            dist_mat = weight*dist_mat

        if args.decoder == 'mst':
            C_x_j = int(np.argmax(np.sum(final_matrix, axis=0)))
            final_matrix[0] = 0
            if root and args.root == 'gold':
                final_matrix[root] = 0
                final_matrix[root, 0] = 1
            if args.root == 'cls':
                final_matrix[C_x_j] = 0
                final_matrix[C_x_j, 0] = 1
            # final_matrix /= np.sum(final_matrix, axis=1, keepdims=True)

            best_heads = chuliu_edmonds(final_matrix)
            trees.append([(i, head) for i, head in enumerate(best_heads)])

        if args.decoder == 'eisner':
            if root and args.root == 'gold':
                final_matrix[root] = 0
                final_matrix[root, 0] = 1
            C_x_j = int(np.argmax(np.sum(final_matrix, axis=0)))
            if args.root == 'cls':
                final_matrix[C_x_j] = 0
                final_matrix[C_x_j, 0] = 1
            else:
                final_matrix[0] = .1
                final_matrix[0, 0] = 0
            final_matrix /= np.sum(final_matrix, axis=1, keepdims=True)
            final_matrix = final_matrix.transpose()

            if args.favor_local:
                final_matrix += dist_mat
                # final_matrix = np.where(final_matrix < 0, 0.0001, final_matrix)
                # for i, row in enumerate(final_matrix):
                #     for j, cell in enumerate(row):
                #         final_matrix[i,j] *= dist_mat[i,j]

            best_heads, _ = decoder.parse_proj(final_matrix)
            for i, head in enumerate(best_heads):
                if head == 0 and i == root:
                    root_found += 1
            trees.append([(i, head) for i, head in enumerate(best_heads)])
        if args.decoder == 'right_chain':
            # trees.append([(root, 0) if i == root else (i, i + 1) for i in range(0, final_matrix.shape[0])])
            trees.append([(i, i + 1) for i in range(0, final_matrix.shape[0])])
        if args.decoder == 'left_chain':
            # trees.append([(root, 0) if i == root else (i, i - 1) for i in range(0, final_matrix.shape[0])])
            trees.append([ (i, i - 1) for i in range(0, final_matrix.shape[0])])
        if args.decoder == 'random':
            random_matrix = np.random.rand(final_matrix.shape[0], final_matrix.shape[0])
            # if root and args.root == 'gold':
            #     random_matrix[root] = 0
            #     random_matrix[root, 0] = 1
            best_heads, _ = decoder.parse_proj(random_matrix)
            trees.append([(i, head) for i, head in enumerate(best_heads)])
        if args.decoder == 'rosa':
            sent_scores = final_matrix.sum(axis=0)[1:] # remove [CLS]
            sent_root = -1
            edges = [(0,0)]
            for rosa_id in range(sent_scores.shape[0]):
                parent = -1
                for potential_parent in range(rosa_id, sent_scores.shape[0]):
                    if sent_scores[rosa_id] < sent_scores[potential_parent]:
                        parent = potential_parent
                        break
                if parent == -1:
                    parent = sent_root
                if parent == -1:
                    sent_root = rosa_id
                edges.append((rosa_id+1, parent+1))
            trees.append(edges)
        if args.decoder == 'clark':
            # final_matrix /= np.sum(final_matrix, axis=1, keepdims=True)
            # final_matrix = final_matrix.transpose()
            heads = [(0, 0)]
            most_imp_word = np.argmax(final_matrix[0])
            for i in range(1, final_matrix.shape[0]):
                row = final_matrix[i]
                head_of_row = np.argmax(row)
                heads.append((i, head_of_row))
            heads[root] = (root, 0)
            trees.append(heads)
    print("found root: ", root_found, len(trees))
    return trees, new_results, deprels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--matrix', default='./results/PUD/bert-base-uncased-False-dist-12.pkl')
    # parser.add_argument('--matrix', default='./results/WSJ10/bert-base-uncased-False-diff.pkl')
    # parser.add_argument('--matrix', default='./results/WSJ10/bert-base-uncased-False-dist-12.pkl')

    # Decoding args
    parser.add_argument('--decoder', default='eisner')
    parser.add_argument('--root', default='gold', help='gold or cls')
    parser.add_argument('--subword', default='first')
    parser.add_argument('--favor_local', action='store_true')

    args = parser.parse_args()
    print(args)
    trees, results, deprels = decoding(args)
    new_evaluation(trees, results)
    # distance_analysis(trees, results)
    # distance_tag_analysis(trees, results, deprels)
    # tag_evaluation(trees, results, deprels)
    # no_punc_evaluation(trees, results, deprels)