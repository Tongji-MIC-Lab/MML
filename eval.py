#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import pearsonr


def loadgt(file_ground_truth=''):
    # Extracting Ground Truth annotation
    with open(file_ground_truth,'r') as f:
        lines = f.read().splitlines()

    names_gt = []
    valence_gt = []
    arousal_gt = []
    for line in lines:
        items = line.split()
        name = items[0]
        ind = name.find(".")
        if ind >=0:
            name = name[0:ind]

        names_gt.append(name)
        valence_gt.append(float(items[1]))
        arousal_gt.append(float(items[2]))
    return valence_gt, arousal_gt

def evalEIMT16(valence, arousal, file_ground_truth):
    valence_gt, arousal_gt = loadgt(file_ground_truth)
    if None != valence:
        mseval = mean_squared_error(valence_gt, valence)
        pccval = pearsonr(valence_gt, valence)[0]
    if None != arousal:
        msearo = mean_squared_error(arousal_gt, arousal)
        pccaro = pearsonr(arousal_gt, arousal)[0]
    return mseval, pccval, msearo, pccaro



