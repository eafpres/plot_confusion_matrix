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
    plt.show()
#

