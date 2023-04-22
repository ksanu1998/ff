import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

ff_type = "linear"

plot_directory = './plots'

if not os.path.exists(plot_directory):
	os.mkdir(plot_directory)

datasets = ["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "CIFAR100"]
dataset_labels = ["MNIST", "Fashion\nMNIST", "SVHN", "CIFAR10", "CIFAR100"]
fnames = []
if not os.path.exists(os.path.join(plot_directory, "e2e")):
	os.mkdir(os.path.join(plot_directory, "e2e"))
plot_directory = os.path.join(plot_directory, "e2e")

for dataset in datasets:
	fnames.append(os.path.join("./content", ff_type, dataset, "e2e_ff.csv"))
layer_1 = []
for fname in fnames:
	print(fname)
	with open(fname) as csvfile:
		count = 0
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
		    if count < 1:
		        count += 1
		        continue
		    layer_1.append(float(row[0])/1000)
		    count += 1

print(layer_1)

fig, axs= plt.subplots(1,1)
axs.set_ylabel('E2E Time (sec)', size=15)
axs.scatter(np.arange(1,6), layer_1)
axs.tick_params(axis='x', which='minor', bottom=False)
plt.minorticks_on() 
plt.grid(which='major', axis='both', linewidth=1)
plt.grid(which='minor', axis='both',linestyle=':', linewidth=0.6)
plt.xticks(np.arange(1,6), size = 15)
axs.set_xticklabels(dataset_labels, rotation=90)
plt.yticks(size = 15)
# plt.title("FC Network with FF", size=15)
axs.set_ylim(0,1500)
fig.set_size_inches(3, 5)
plt.savefig(os.path.join(plot_directory, "e2e.png"), format='png', bbox_inches='tight', dpi=600)