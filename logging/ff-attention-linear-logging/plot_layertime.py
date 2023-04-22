import csv
import matplotlib.pyplot as plt
import numpy as np
import os

dataset = "SVHN"
ff_type = "attention"
epochs = 60
num_layers = 4


plot_directory = "./plots"
if not os.path.exists(plot_directory):
	os.mkdir(plot_directory)
plot_directory = os.path.join("./plots", ff_type)
if not os.path.exists(plot_directory):
	os.mkdir(plot_directory)
	
plot_subdirectory = 'layer_time'
if not os.path.exists(os.path.join(plot_directory, plot_subdirectory)):
	os.mkdir(os.path.join(plot_directory, plot_subdirectory))
plot_directory = os.path.join(plot_directory, plot_subdirectory)

if not os.path.exists(os.path.join(plot_directory, dataset)):
	os.mkdir(os.path.join(plot_directory, dataset))

fname = os.path.join("./", "content", ff_type, dataset, "layer_ff.csv")

layers = []
with open(fname) as csvfile:
	count = 0
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
	    if count < 1:
	        count += 1
	        continue
	    layers.append(float(row[1])/1000)
	    count += 1
print(layers)
fig, axs= plt.subplots(1,1)
axs.set_xlabel('Layer #', size=15)
axs.plot(layers)
axs.tick_params(axis='x', which='minor', bottom=False)
plt.minorticks_on() 
plt.grid(which='major', axis='both', linewidth=1)
plt.grid(which='minor', axis='both',linestyle=':', linewidth=0.6)
plt.xticks(size = 15)
plt.yticks(size = 15)
# plt.legend(fontsize=12)
if dataset == "MNIST":
	axs.set_ylabel('Layer Time\n(sec)', size=15)
else:
	axs.yaxis.set_major_formatter(plt.NullFormatter())
axs.set_ylim(0,300)
axs.set_xlim(0,num_layers-1)
fig.set_size_inches(3, 5)
plt.savefig(os.path.join(plot_directory, dataset, "layer_time.png"), format='png', bbox_inches='tight', dpi=600)