# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from   matplotlib import colors
import seaborn as sns

import json

"""Loading JSON data:"""

base_path='/content/drive/MyDrive/Google_AI_Studio/ARCAGI2025/'

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data



"""Reading files:"""

training_challenges   = load_json(base_path +'arc-agi_training_challenges.json')
training_solutions    = load_json(base_path +'arc-agi_training_solutions.json')

evaluation_challenges = load_json(base_path +'arc-agi_evaluation_challenges.json')
evaluation_solutions  = load_json(base_path +'arc-agi_evaluation_solutions.json')

"""### Function to plot input/output pairs of a task"""

# 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

plt.figure(figsize=(3, 1), dpi=150)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.tick_params(axis='x', color='r', length=0, grid_color='none')

plt.show()

def plot_task(task, task_solutions, i, t, size=2.5, w1=0.9):
    t=list(training_challenges)[i]
    titleSize=16
    num_train = len(task['train'])
    num_test  = len(task['test'])

    wn=num_train+num_test
    fig, axs  = plt.subplots(2, wn, figsize=(size*wn,2*size))
    plt.suptitle(f'Task #{i}, {t}', fontsize=titleSize, fontweight='bold', y=1, color = '#eeeeee')

    '''train:'''
    for j in range(num_train):
        plot_one(axs[0, j], j,task, 'train', 'input',  w=w1)
        plot_one(axs[1, j], j,task, 'train', 'output', w=w1)

    '''test:'''
    for k in range(num_test):
        plot_one(axs[0, j+k+1], k, task, 'test', 'input', w=w1)
        task['test'][k]['output'] = task_solutions[k]
        plot_one(axs[1, j+k+1], k, task, 'test', 'output', w=w1)

    axs[1, j+1].set_xticklabels([])
    axs[1, j+1].set_yticklabels([])
    axs[1, j+1] = plt.figure(1).add_subplot(111)
    axs[1, j+1].set_xlim([0, wn])

    '''Separators:'''
    colorSeparator = 'white'
    for m in range(1, wn):
        axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color = colorSeparator)
    axs[1, j+1].plot([num_train,num_train],[0,1],'-', linewidth=3, color = colorSeparator)

    axs[1, j+1].axis("off")

    '''Frame and background:'''
    fig.patch.set_linewidth(5) #widthframe
    fig.patch.set_edgecolor('black') #colorframe
    fig.patch.set_facecolor('#444444') #background

    plt.tight_layout()

    print(f'#{i}, {t}') # for fast and convinience search
    plt.show()

def plot_one(ax, i, task, train_or_test, input_or_output, solution=None, w=0.8):
    fs=12
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    #ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])

    '''Grid:'''
    ax.grid(visible= True, which = 'both', color = '#666666', linewidth = w)

    ax.tick_params(axis='both', color='none', length=0)

    '''sub title:'''
    ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs, color = '#dddddd')

"""# <div  style="color:white; border:lightgreen solid;  font-weight:bold; font-size:120%; text-align:center;padding:12.0px; background:black">2. VISUALIZATION TRAINING SET</div>

## 000-100 tasks
"""

for i in range(0, 100):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t]
    plot_task(task,  task_solution, i, t)

"""## 100-200 tasks"""

for i in range(100, 200):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t]
    plot_task(task,  task_solution, i, t)

"""## 200-300 tasks"""

for i in range(200, 300):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t]
    plot_task(task,  task_solution, i, t)

"""## 300-400 tasks"""

for i in range(300, 400):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t]
    plot_task(task,  task_solution, i, t)

"""## 400-500 tasks"""

for i in range(400, 500):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t]
    plot_task(task,  task_solution, i, t)

"""## 500-600 tasks"""

for i in range(500, 600):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t]
    plot_task(task,  task_solution, i, t)

"""## 600-700 tasks"""

for i in range(600, 700):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t]
    plot_task(task,  task_solution, i, t)

"""## 700-800 tasks"""

for i in range(700, 800):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t]
    plot_task(task,  task_solution, i, t)

"""## 800-900 tasks"""

for i in range(800, 900):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t]
    plot_task(task,  task_solution, i, t)

"""## 900-1000 tasks"""

for i in range(900, 1000):
    if i not in []:
        t=list(training_challenges)[i]
        task=training_challenges[t]
        task_solution = training_solutions[t]
        plot_task(task,  task_solution, i, t)

"""# <div  style="color:white; border:lightgreen solid;  font-weight:bold; font-size:120%; text-align:center;padding:12.0px; background:black">2. VISUALIZATION EVALUATION SET</div>

## 000-120 tasks
"""

for i in range(0, 120):
    t=list(evaluation_challenges)[i]
    task=evaluation_challenges[t]
    task_solution = evaluation_solutions[t]
    plot_task(task,  task_solution, i, t)

