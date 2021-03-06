# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:19:21 2020

@author: Blaine Bateman
"""
#
#%% libraries
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
#
#%% plot confusion matrix
#
def plot_confusion_matrix(cm, title, classes):
#
# calculate accuracies
#
    acc = []
    title = title + '\n'
    for i in range(cm.shape[0]):
        acc.append(cm[i, i] / np.sum(cm[:, i]))
        title = title + str(round(acc[i], 2)) + '    '
#
# define the grid
#
    grid_size = cm.shape[0]
#
# create values to plot as 1s on the diagonal, 0s elsewhere
#
    data = np.zeros((grid_size, grid_size))
    np.fill_diagonal(data, 1)
#
# initialize plot
#
    fig, ax = plt.subplots(1, 1, tight_layout = True)
#
# make color map with number of elements equal to number of squares
#
    grid_colors = ['lightgrey' for i in range(grid_size * grid_size + 1)]
    grid_colors[grid_size * grid_size] = 'lightgreen'
    my_cmap = colors.ListedColormap(grid_colors)
#
# draw the grid
#
    for x in range(grid_size + 1):
        ax.axhline(x, lw = 1, color = 'blue', zorder = 1)
        ax.axvline(x, lw = 1, color = 'blue', zorder = 1)
#
# color the squares
#
    ax.imshow(data,
              interpolation = 'none',
              cmap = my_cmap,
              extent = [0 - 0.01, grid_size + 0.01, 0 - 0.01, grid_size + 0.01],
              zorder = 0)
#
# axis labels
#
    tick_pos = np.arange(0.5, grid_size, 1)
    x_labels = classes
    y_labels = classes[::-1]
    ax.set_xlabel('Actual Label')
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(x_labels, fontsize = 7)
    ax.set_ylabel('Predicted Label')
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(y_labels, rotation = 'vertical', va = 'center', fontsize = 7)
#
# label squares
#
    for x in range(grid_size):
        for y in range(grid_size):
            ax.text(tick_pos[x], tick_pos[grid_size - y - 1],
                    str(cm[y, x]),
                    ha = 'center')
#
# add title
#
    plt.suptitle('Predictions from model')
    ax.set_title('\n' + title, fontsize = 10)
#
#%% example code
#
#%% multi-class example
#
data = pd.read_csv('iris.data.txt')
iris_label_enc = OrdinalEncoder().fit(pd.DataFrame(data.iloc[:, -1]))
iris_classes = iris_label_enc.transform(np.reshape(pd.DataFrame(data.iloc[:, -1]), (-1, 1)))
iris_labels = list(iris_label_enc.categories_[0])
model = OneVsRestClassifier(LogisticRegression()).fit(data.iloc[:, :-1], iris_classes.ravel())
iris_preds = model.predict(data.iloc[:, :-1])
cm = confusion_matrix(iris_classes.ravel(), iris_preds)
plot_confusion_matrix(cm, 'iris confusion matrix', iris_labels)
#
#%% multi-label example
#
rows = 100
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
label_cols = pd.DataFrame(columns = labels)
for i in range(rows):
    for column in range(len(labels)):
        if np.random.randint(0, 2) == 1:
            label_cols.loc[i, labels[column]] = labels[column]
        else:
            label_cols.loc[i, labels[column]] = 'NA'
errors = [1, 1, 1, 1, 1, 1, 0]
predicted_cols = pd.DataFrame(columns = labels)
for i in range(rows):
    for column in range(len(labels)):
        predicted_cols.loc[i, labels[column]] = np.random.choice(errors)
for i in range(rows):
    for position in range(len(labels)):
        if predicted_cols.loc[i, labels[position]] == 1:
            predicted_cols.loc[i, labels[position]] = label_cols.loc[i, labels[position]]
        else:
            predicted_cols.loc[i, labels[position]] = np.random.choice([labels[position], 'NA'])
for i in range(len(labels)):
    cm = confusion_matrix(predicted_cols.loc[:, labels[i]], label_cols.loc[:, labels[i]], labels = [labels[i], 'NA'])
    plot_confusion_matrix(cm, 'confusion matrix class ' + labels[i], classes = [labels[i], 'NA'])
