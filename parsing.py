import argparse
from tqdm import tqdm
import numpy as np
import pickle

from dependency import _evaluation as dep_eval
from dependency.dep_parsing import decoding as dep_parsing
from discourse.dis_parsing import decoding as dis_parsing
from discourse import evaluation as dis_eval
from discourse import distance_evaluation as dis_eval_per_distance
from constituency.con_parsing import decoding as con_parsing
from constituency.con_parsing import constituent_evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--matrix', default='./results/discourse/bert-dist-SciDTB-last.pkl')
    parser.add_argument('--probe', default='discourse', choices=['dependency', 'constituency', 'discourse'])
    parser.add_argument('--decoder', default='eisner', choices=['eisner', 'cle', 'right_chain',
                                                                'top_down', 'mart', 'right_branching', 'left_branching'])
    parser.add_argument('--subword', default='avg', choices=['first', 'avg', 'max'])
    parser.add_argument('--root', default='gold', help='use gold root as init')

    args = parser.parse_args()
    print(args)

    if args.probe == 'dependency':
        trees, results, deprels = dep_parsing(args)
        dep_eval(trees, results)
    elif args.probe == 'constituency':
        trees, results = con_parsing(args)
        constituent_evaluation(trees, results)
    elif args.probe == 'discourse':
        trees, gold_trees, deprels = dis_parsing(args)
        dis_eval(trees, gold_trees)
        dis_eval_per_distance(trees, gold_trees)

