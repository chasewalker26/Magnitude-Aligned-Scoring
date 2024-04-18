import numpy as np
from scipy.stats import spearmanr

# Sanity Checks for Saliency Metrics
# https://arxiv.org/pdf/1912.01451v1.pdf

def interval_distance(a, b):
    return (a - b) ** 2

# https://github.com/grrrr/krippendorff-alpha/blob/master/krippendorff_alpha.py
def krippendorff_alpha(rankings):  
    n = rankings.size

    Do = 0
    for unit in rankings:
        Du = 0
        for v in unit:
            Du += np.sum(interval_distance(unit, v)) 
        Do += Du / len(unit)
    Do = Do / n

    De = 0
    for unit_a in rankings:
        for unit_b in rankings:
            for v in unit_b:
                De += np.sum(interval_distance(unit_a, v))
    De = De / (n * (n - 1))

    return 1 - (Do / De)

# rankings: N x M 
# Each row represents the ranking of M saliency maps for an image by a metric score
# Each position has a number 0 - M - 1 which holds the rank of the saliency map at the index
# The rankings array holds this data for N images total
# Use this to measure how well each metric consistently ranks saliency maps individually
def inter_rater_reliability(rankings):
    return krippendorff_alpha(rankings.T)

# Two sets of rankings N x M from two differnent metrics
# Each row represents the ranking of M saliency maps for an image by a metric score
# Each position has a number 0 - M - 1 which holds the rank of the saliency map at the index
# The rankings array holds this data for N images total
# Use this to measure RISE ins and del vs MAS ins and del internal consistency
def internal_consistency_reliability(rankings_a, rankings_b):
    return np.mean(spearmanr(rankings_a, rankings_b, axis = 1).correlation)

# Each score hold N scores of one saliency method for N images
# Use this to measure how well one metric scores two saliency methods across a dataset
def inter_method_reliability(scores_a, scores_b):
    return spearmanr(scores_a, scores_b).correlation
