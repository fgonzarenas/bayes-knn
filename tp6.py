#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk

def parse(filename):
    with open(filename) as file:
        X = []
        y = []
        
        for line in file:
            line = line.strip('\n').split(',')
            X.append(line[:-1])
            y.append(line[-1])
            
    return np.array(X), np.array(y)


if __name__ == '__main__':
    filename = 'data/data2.csv'
    X, y = parse(filename)
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size = 0.2)