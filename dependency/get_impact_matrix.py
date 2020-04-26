from transformers import BertModel, BertTokenizer
import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from utils import ConllUDataset
import os
from dependency import match_tokenized_to_untokenized


def get_all_subword_id(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
    return id_for_all_subwords


def main(args, model, tokenizer):
    dataset = ConllUDataset('./data/'+args.dataset+'.conllu')

    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    model.eval()

    out = []
    for line in tqdm(dataset.tokens[1:]):
        sentence = [x.form.lower() for x in line][1:]

        tokenized_text = tokenizer.tokenize(' '.join(sentence))
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        mapping = match_tokenized_to_untokenized(tokenized_text, sentence)

        # 1. Generate mask indices
        matrix_as_list = []
        for i in range(0, len(tokenized_text)):
            id_for_all_i_tokens = get_all_subword_id(mapping, i)
            tmp_indexed_tokens = list(indexed_tokens)
            for tmp_id in id_for_all_i_tokens:
                if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
                    tmp_indexed_tokens[tmp_id] = mask_id
            one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
            for j in range(0, len(tokenized_text)):
                id_for_all_j_tokens = get_all_subword_id(mapping, j)
                for tmp_id in id_for_all_j_tokens:
                    if mapping[tmp_id] != -1:
                        one_batch[j][tmp_id] = mask_id

            # 2. Convert one batch to PyTorch tensors
            tokens_tensor = torch.tensor(one_batch)
            if args.cuda:
                tokens_tensor = tokens_tensor.to('cuda')
                model.to('cuda')

            # 3. get all hidden states for one batch
            with torch.no_grad():
                model_outputs = model(tokens_tensor)
                last_layer = model_outputs[0]

            # 4. get hidden states for word_i in one batch
            if args.cuda:
                hidden_states_for_token_i = last_layer[:, i, :].cpu().numpy()
            else:
                hidden_states_for_token_i = last_layer[:, i, :].numpy()
            matrix_as_list.append(hidden_states_for_token_i)

        init_matrix = np.zeros((len(tokenized_text), len(tokenized_text)))
        for i, hidden_states in enumerate(matrix_as_list):
            base_state = hidden_states[i]
            for j, state in enumerate(hidden_states):
                if args.metric == 'dist':
                    init_matrix[i][j] = np.linalg.norm(base_state - state)
                if args.metric == 'cos':
                    init_matrix[i][j] = np.dot(base_state, state) / (
                                np.linalg.norm(base_state) * np.linalg.norm(state))
                # design your own metric, such KL
        out.append((line, tokenized_text, init_matrix))

    with open(args.output_file, 'wb') as fout:
        pickle.dump(out, fout)
        fout.close()


if __name__ == '__main__':
    MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer, 'bert-base-uncased'),
    }
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument("--model_type", default='bert', type=str)

    # Data args
    parser.add_argument('--dataset', default='WSJ10')
    parser.add_argument('--output_dir', default='./results/')

    # Matrix args
    parser.add_argument('--metric', default='dist')

    # Cuda
    parser.add_argument('--cuda', action='store_true')

    args = parser.parse_args()

    model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[args.model_type]

    args.output_dir = args.output_dir + args.dataset
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_file = args.output_dir + '/{}-{}-last_layer.pkl'.format(args.model_type, args.metric)

    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
    print(args)
    main(args, model, tokenizer)

