import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
sns.set_style('whitegrid') # Use seaborn defaults for plotting

parser = argparse.ArgumentParser(description="plot accuracy")
parser.add_argument('--results_file', type=str, default=None)
parser.add_argument('--plot_title', type=str, default=None)
args = parser.parse_args()

results = [line.split() for line in open(args.results_file, 'r').readlines()]

results_verb = {} 
results_noun = {} 

results_verb_total = {}
results_noun_total = {}

results_verb_TD = [] 
results_verb_FA = []
results_noun_TD = [] 
results_noun_FA = []

for line in results:
    results_verb[line[0]] = 0
    results_noun[line[4]] = 0
    results_verb_total[line[0]] = 0
    results_noun_total[line[4]] = 0
for line in results:
    results_verb_TD.append(float(line[2]))
    results_verb_FA.append(float(line[3]))
    results_noun_TD.append(float(line[6]))
    results_noun_FA.append(float(line[7]))

    if int(line[1])<1: # top position
        results_verb[line[0]] += 100        
    else: 
        results_verb[line[0]] += 50-5*int(line[1])

    if int(line[5])<1: 
        results_noun[line[4]] += 100 # top position
    else:
        results_noun[line[4]] += 50-5*int(line[5])

    results_verb_total[line[0]] += 1
    results_noun_total[line[4]] += 1

# prepare data for TD and FA rate vs probability threshold curves 

verb_TD = []
verb_FA = []
noun_TD = []
noun_FA = []
prob_TR = [] # probability threshold

for prob in np.arange(0.0, 1.0, 0.1):
    prob_TR.append(prob)
    verb_TD.append(100*len([td for td in results_verb_TD if td>prob])/len(results_verb_TD))
    verb_FA.append(100*len([fa for fa in results_verb_FA if fa>prob])/len(results_verb_FA))
    noun_TD.append(100*len([td for td in results_noun_TD if td>prob])/len(results_noun_TD))
    noun_FA.append(100*len([fa for fa in results_noun_FA if fa>prob])/len(results_noun_FA))

# prepare data for detection rates

verb_labels = []
verb_values = []

for key in results_verb.keys():
    verb_labels.append(key)
    verb_values.append(results_verb[key]/results_verb_total[key])

noun_labels = []
noun_values = []

for key in results_noun.keys():
    noun_labels.append(key)
    noun_values.append(results_noun[key]/results_noun_total[key])

fig, axs = plt.subplots(1, 4, constrained_layout=True)
axs[0].set_ylim(0, 101)
axs[0].set_xlabel('Verb')
axs[0].set_ylabel('Verb detection rate')
axs[0].bar(verb_labels, verb_values)
axs[1].set_ylim(0, 101)
axs[1].set_xlabel('Noun')
axs[1].set_ylabel('Noun detection rate')
axs[1].bar(noun_labels, noun_values)
axs[2].set_xlabel('Probability Threshold')
axs[2].set_ylim(0, 101)
axs[2].set_ylabel('Verb TD (green) and FA (red) rates')
axs[2].plot(prob_TR, verb_TD, 'g')
axs[2].plot(prob_TR, verb_FA, 'r')
axs[3].set_xlabel('Probability Threshold')
axs[3].set_ylim(0, 101)
axs[3].set_ylabel('Noun TD (green) and FA (red) rates')
axs[3].plot(prob_TR, noun_TD, 'g')
axs[3].plot(prob_TR, noun_FA, 'r')
plt.suptitle(args.plot_title)
plt.show()

