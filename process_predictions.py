import argparse
import os
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--score_paths', nargs='+', type=str, help='Path to prediction files')
    parser.add_argument(
        '--maxes',
        action='store_true',
        help='Display max statistics about predictions')
    parser.add_argument(
        '--means',
        action='store_true',
        help='Display statistics about mean of abs vals')
    parser.add_argument(
        '--abs_means',
        action='store_true',
        help='Display absolute mean statistics about predictions')
    parser.add_argument(
        '--values',
        action='store_true',
        help=
        'Display statistics about prediction score samples (accross all samples)'
    )
    args = parser.parse_args()

    all_scores = []
    for score_path in args.score_paths:
        with open(score_path, 'rb') as f:
            scores = pickle.load(f)
            print("Got {} scores from {}".format(len(scores), score_path))
            all_scores.append(scores)
    all_means = []
    all_maxes = []
    all_abs_means = []
    all_values = []
    for score in tqdm(all_scores, desc='checkpoint'):
        maxes = []
        means = []
        abs_means = []
        values = []
        for pred in tqdm(score.values(), desc='sample'):
            pred = pred.cpu()
            if args.maxes:
                pred_max = pred.max()
                maxes.append(pred_max)
            if args.means:
                pred_mean = pred.mean()
                means.append(pred_mean)
            if args.abs_means:
                pred_abs_mean = pred.abs().mean()
                abs_means.append(pred_abs_mean)
            if args.values:
                values.append(pred.numpy().tolist())

        # Flatten scores
        values = [val for vals in values for val in vals]

        all_maxes.append(maxes)
        all_means.append(means)
        all_abs_means.append(abs_means)
        all_values.append(values)
    labels = [
        os.path.basename(os.path.dirname(score)) for score in args.score_paths
    ]

    # Plot mean scores
    if args.means:
        plt.hist(all_means, 50, label=labels)
        plt.title('means')
        plt.legend()
        plt.show()

    # Plot abs mean scores
    if args.abs_means:
        plt.hist(all_abs_means, 50, label=labels)
        plt.title('Mean of absolute values')
        plt.legend()
        plt.show()

    # Plot predictions histogram
    if args.values:
        plt.hist(all_values, 50, label=labels)
        plt.title('Histogram of scores')
        plt.legend()
        plt.show()

    # Plot max scores
    if args.maxes:
        plt.hist(all_maxes, 50, label=labels)
        plt.title('maxes')
        plt.legend()
        plt.show()
    import pdb
    pdb.set_trace()
