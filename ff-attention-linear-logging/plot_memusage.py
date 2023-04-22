import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


dataset = "MNIST"
ff_type = "linear"

plot_directory = "./plots"
if not os.path.exists(plot_directory):
	os.mkdir(plot_directory)
plot_directory = os.path.join("./plots", ff_type)
if not os.path.exists(plot_directory):
	os.mkdir(plot_directory)

plot_subdirectory = 'mem_usage'
if not os.path.exists(os.path.join(plot_directory, plot_subdirectory)):
	os.mkdir(os.path.join(plot_directory, plot_subdirectory))
plot_directory = os.path.join(plot_directory, plot_subdirectory)

if not os.path.exists(os.path.join(plot_directory, dataset)):
	os.mkdir(os.path.join(plot_directory, dataset))
fname = os.path.join("./", "content", ff_type, dataset, "nvidia_smi.csv")

values = []
with open(fname) as csvfile:
	count = 0
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		if len(row) != 0:
			values.append(100*float(row[4].replace('MiB','').replace(' ',''))/float(row[6].replace('MiB','').replace(' ','')))
		count += 1

# print(values)
fig, axs= plt.subplots(1,1)
# axs.set_xlabel(dataset, size=15)
# axs.plot(layer_1, label ='Layer 1')
# axs.plot(layer_2, label ='Layer 2')
# violinplot = sns.violinplot(data=values, ax=axs, linewidth=1, edgecolor='white')
barplot = sns.barplot(data=values, ax=axs, linewidth=1, edgecolor='white', estimator=np.mean)

axs.tick_params(axis='x', which='minor', bottom=False)
plt.minorticks_on() 
plt.grid(which='major', axis='both', linewidth=1)
plt.grid(which='minor', axis='both',linestyle=':', linewidth=0.6)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend([dataset], fontsize=15)
axs.xaxis.set_major_formatter(plt.NullFormatter())
axs.set_axisbelow(True)
axs.set_ylabel('Memory Usage (%)', size=15)
'''
if dataset == "MNIST":
	axs.set_ylabel('GPU Utilization (%)', size=15)
else:
	axs.yaxis.set_major_formatter(plt.NullFormatter())
'''
# plt.title("FC Network with FF using " + dataset, size=15)
axs.set_ylim(0,100)
# axs.set_xlim(0,1000)
fig.set_size_inches(3, 5)
plt.savefig(os.path.join(plot_directory, dataset, "mem_usage.png"), format='png', bbox_inches='tight', dpi=600)
