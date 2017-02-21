"""
    performance.py
    
    Evaluate performance of `PFR` object detector
"""

import os
import ujson as json
import numpy as np
import pandas as pd
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from subprocess import Popen

def show_plot(p='./tmp.png'):
    plt.savefig(p)
    plt.close()
    Popen(('rsub %s' % p).split())


# --
# Load anns

class_ = "vehicle"
label_ = "combat_vehicle"
iter_ = "final"

ann_path = '/home/bjohnson/projects/py-faster-rcnn/custom-tools/output/eval-f/%s.json' % label_
anns = json.load(open(ann_path))

all_anns = set([ann['filename'] for ann in anns])
pos_anns = set([ann['filename'] for ann in anns if len(ann['annotations']) > 0])

# --
# Load scores

score_path = './results/f-results-yolo-custom_%s' % str(iter_)
scores = pd.read_csv(score_path, sep='\t', header=None)

# Subset to class of interest
assert((scores[1] == label_).sum() > 0)
scores = scores[scores[1] == label_]

# Rename
scores = scores[[0,2]]
scores.columns = ('file', 'score')
scores.file = scores.file.apply(os.path.basename)

# Add labels
to_add = pd.DataFrame({"file" : list(all_anns)})
scores = pd.merge(scores, to_add, how='outer')
scores.score[scores.score.isnull()] = -1

scores['ann'] = scores.file.isin(all_anns)
scores['pos'] = scores.file.isin(pos_anns)

scores = scores.sort_values('score', ascending=False).reset_index(drop=True)
scores = scores.loc[scores.file.drop_duplicates().index].reset_index(drop=True)

# --
# Plot performance

# assert(scores.ann.mean() == 1)
sub = scores[scores.ann].reset_index(drop=True) # Removing reset_index gives you maximally pessimistic estimate

# Precision/recall
prec, recall, _ = metrics.precision_recall_curve(sub.pos, sub.score)
_ = plt.plot(recall * sub.pos.sum(), prec)
_ = plt.xlabel('N positives found')
_ = plt.ylabel('Precision')
_ = plt.ylim(0, 1.05)
_ = plt.xlim(0, sub.pos.sum() * 1.05)
_ = plt.title('darknet Object Detection: %s' % label_)
show_plot(p='./figures/darknet-%s-pr.png' % label_)

# Histogram of sub
_ = plt.hist(sub.score[sub.pos], 50, alpha=0.3, label='pos', color='red')
_ = plt.hist(sub.score[~sub.pos], 50, alpha=0.3, label='neg', color='blue')
_ = plt.xlabel('Score')
_ = plt.ylabel('Count')
_ = plt.title("darknet Object Detection: %s -- histogram of sub" % label_)
_ = plt.legend(loc='upper right')
show_plot(p='./figures/darknet-%s-hist.png' % label_)

# Recall at K plot
_ = plt.plot(np.cumsum(sub.pos))
_ = plt.plot(np.arange(500), alpha=0.2)
_ = plt.ylim(0, sub.pos.sum() * 1.05)
_ = plt.xlabel('K')
_ = plt.ylabel('N positive')
_ = plt.title("darknet Object Detection: %s -- recall @ k" % label_)
show_plot(p='./figures/darknet-%s-rak.png' % label_)
