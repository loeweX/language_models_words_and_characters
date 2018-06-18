###############################################################################
#
# Evaluating the results, print values and plot graphs
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import math
from os import listdir
from os.path import isfile, join

plt.rcParams.update({'font.size': 22})

OUTPUT_DIR = './output/'
eval_files = [f for f in listdir(OUTPUT_DIR) if isfile(join(OUTPUT_DIR, f))]

counter = 0
width = 0.25

fig, ax = plt.subplots(figsize=(15, 5))
fig, loss_plot = plt.subplots(figsize=(7.5, 5))

loss_train_char = []
loss_val_char = []
loss_train_sub = []
loss_val_sub = []
loss_train_word = []
loss_val_word = []

for num, name in enumerate(sorted(eval_files)):

    if name[-1] == 'z':
        print(name)
    else:
        continue

    file = np.load(OUTPUT_DIR + name)

    loss_train=file['loss_train']
    loss_val=file['loss_val']
    loss_test=file['loss_test']
    bpc_train=file['bpc_train']
    bpc_val=file['bpc_val']
    bpc_test=file['bpc_test']
    ppl_train=file['ppl_train']
    ppl_val=file['ppl_val']
    ppl_test=file['ppl_test']

    train_avg_len = file['train_avg_len']
    valid_avg_len = file['valid_avg_len']
    test_avg_len = file['test_avg_len']
    num_param = file['num_param']
    ntokens = file['ntokens']
    learning_rates=file['learning_rates']

    if name[-5] == 'r':  # char model comes always first for each dataset (alphabetical order horray!)
        avg_len = test_avg_len

    print('Base BPC for Testing: ', (bpc_test[0]*test_avg_len)/(avg_len))
    print('Best BPC for Testing: ', (min(bpc_test)*test_avg_len)/(avg_len))
    print('Base Perplexity for Testing: ', ppl_test[0])
    print('Best Perplexity for Testing: ', min(ppl_test))

    print('Relative Improvement Base to Best: ', ppl_test[0] / min(ppl_test))

    print('Average word length test: ', test_avg_len)
    print('Number of parameters in the model: ', num_param)
    print('Number of different words/subwords/characters in the dataset: ', ntokens)
    #print('Learning rates: ', learning_rates)
    print(ppl_val)
    print()

    plt.figure()#num
    plt.plot(loss_train, color='blue')
    plt.plot(loss_val, color='red')
    plt.title(name)

    if name[3] == 'P': #penntree
        counter = 1
        #if name[-9] == 'g':  # big model
        #    ax.bar(counter + width, min(ppl_test), width, color='C3', label='big')
        #    ax.bar(counter + width, ppl_test[0], width, color='C3', alpha=0.5)
        if name[-5] == 'r': #char model
            ax.bar(counter, min(ppl_test), width, label='character', color='C0')

            #ax.plot([counter-0.5*width, counter+0.5*width], [167, 167], "k--")
        elif name[-5] == 'b': #subword model
            ax.bar(counter+width, min(ppl_test), width, label='subword', color='C3')
        elif name[-5] == 'd': #word model
            ax.bar(counter+2*width, min(ppl_test), width, label='word', color='C2')
            #ax.plot([counter+1.5*width, counter+2.5*width], [138, 138], "k--", label='Graves')
    elif name[3] == 'E': #english
        counter = 2
    elif name[3] == 'G': #german
        counter = 3
    else: #turkish
        counter = 4

    if name[-5] == 'r':  # char model
        ax.bar(counter, min(ppl_test), width, color='C0')
        ax.bar(counter, ppl_test[0], width, color='C0', alpha=0.5)

        loss_train_char.append(bpc_train)
        loss_val_char.append(bpc_val)
    elif name[-5] == 'b':  # subword model
        ax.bar(counter + width, min(ppl_test), width, color='C3')
        ax.bar(counter + width, ppl_test[0], width, color='C3', alpha=0.5)

        loss_train_sub.append((bpc_train*train_avg_len)/avg_len)
        loss_val_sub.append((bpc_val*valid_avg_len)/avg_len)
    elif name[-5] == 'd':  # word model
        ax.bar(counter + 2 * width, min(ppl_test), width, color='C2')
        ax.bar(counter + 2 * width, ppl_test[0], width, color='C2', alpha=0.5)

        loss_train_word.append((bpc_train*train_avg_len)/avg_len)
        loss_val_word.append((bpc_val*valid_avg_len)/avg_len)

    #if name[-9] == 'g':  # big model
    #    ax.bar(counter + width, min(ppl_test), width, color='C3')
    #    ax.bar(counter + width, ppl_test[0], width, color='C3', alpha=0.5)


### training plot with BPC means and stds
loss_train_char = np.array(loss_train_char)
loss_val_char = np.array(loss_val_char)
mean_loss_train_char = np.mean(loss_train_char, axis=0)
mean_loss_val_char = np.mean(loss_val_char, axis=0)
std_loss_train_char = np.std(loss_train_char, axis=0)
std_loss_val_char = np.std(loss_val_char, axis=0)

loss_train_sub = np.array(loss_train_sub)
loss_val_sub = np.array(loss_val_sub)
mean_loss_train_sub = np.mean(loss_train_sub, axis=0)
mean_loss_val_sub = np.mean(loss_val_sub, axis=0)
std_loss_train_sub = np.std(loss_train_sub, axis=0)
std_loss_val_sub = np.std(loss_val_sub, axis=0)

loss_train_word = np.array(loss_train_word)
loss_val_word = np.array(loss_val_word)
mean_loss_train_word = np.mean(loss_train_word, axis=0)
mean_loss_val_word = np.mean(loss_val_word, axis=0)
std_loss_train_word = np.std(loss_train_word, axis=0)
std_loss_val_word = np.std(loss_val_word, axis=0)

print(mean_loss_val_word)
print(mean_loss_val_sub)
print(mean_loss_val_char)

x = np.linspace(0, 20, 20)
#loss_plot.plot(mean_loss_train_char, label='char train')
loss_plot.plot(x, mean_loss_val_char, color='C0', label='character', linewidth=2.0)
#loss_plot.fill_between(x, mean_loss_train_char - std_loss_train_char, mean_loss_train_char + std_loss_train_char, color='C0', alpha=0.5)
loss_plot.fill_between(x, mean_loss_val_char - std_loss_val_char, mean_loss_val_char + std_loss_val_char, color='C0', alpha=0.2)

x = np.linspace(0, 15, 15)
#loss_plot.plot(mean_loss_train_sub, color='C3', label='sub train')
loss_plot.plot(x, mean_loss_val_sub, color='C3', label='subword',linewidth=2.0)
#loss_plot.fill_between(x, mean_loss_train_sub - std_loss_train_sub, mean_loss_train_sub + std_loss_train_sub, color='C3', alpha=0.5)
loss_plot.fill_between(x, mean_loss_val_sub - std_loss_val_sub, mean_loss_val_sub + std_loss_val_sub, color='C3', alpha=0.2)

#loss_plot.plot(mean_loss_train_word, color='C2', label='word train')
loss_plot.plot(x, mean_loss_val_word, color='C2', label='word', linewidth=2.0)
#loss_plot.fill_between(x, mean_loss_train_word - std_loss_train_word, mean_loss_train_word + std_loss_train_word, color='C2', alpha=0.5)
loss_plot.fill_between(x, mean_loss_val_word - std_loss_val_word, mean_loss_val_word + std_loss_val_word, color='C2', alpha=0.2)

loss_plot.set_xlabel('Epochs')
loss_plot.set_ylabel('BPC on validation')
loss_plot.legend()


##bar plot for perplexities
ind = np.arange(counter)
#handles, labels = ax.get_legend_handles_labels()
#order = [1,2,3,0] #ordering of the legend entries
#ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
ax.legend()
ax.set_ylabel('Perplexity')
ax.set_xticks(ind + width + 1)
ax.set_xticklabels(('Penn Treebank', 'English', 'German', 'Turkish'))

plt.show()

