import csv
import matplotlib.pyplot as plt
import numpy as np
import os

dataset = "SVHN"
ff_type = "backprop"
epochs = 20
num_layers = 1 # DO NOT CHANGE THIS or bad things will happen!!


plot_directory = "./plots"
if not os.path.exists(plot_directory):
	os.mkdir(plot_directory)
plot_directory = os.path.join("./plots", ff_type)
if not os.path.exists(plot_directory):
	os.mkdir(plot_directory)
	
plot_subdirectory = 'epoch_time'
if not os.path.exists(os.path.join(plot_directory, plot_subdirectory)):
	os.mkdir(os.path.join(plot_directory, plot_subdirectory))
plot_directory = os.path.join(plot_directory, plot_subdirectory)

if not os.path.exists(os.path.join(plot_directory, dataset)):
	os.mkdir(os.path.join(plot_directory, dataset))

fname = os.path.join("./", "content", ff_type, dataset, "epoch_linear_ff.csv")

layers = [[] for i in range(num_layers)]
layer_count = 0
with open(fname) as csvfile:
	count = 0
	reader = csv.reader(csvfile, delimiter=',')
	prev_epoch_count = 0
	epoch_count = 0
	for row in reader:
	    if count < 1:
	        count += 1
	        continue
	    epoch_count = int(row[0])
	    layers[layer_count].append(float(row[1])/1000)
	    if epoch_count == epochs - 1:
	    	layer_count += 1
	    count += 1
print(layers)
# if ff_type == 'attention':

# 	fname = os.path.join("./", "content", ff_type, dataset, "epoch_attention_ff.csv")
# 	with open(fname) as csvfile:
# 		count = 0
# 		reader = csv.reader(csvfile, delimiter=',')
# 		prev_epoch_count = 0
# 		epoch_count = 0
# 		val_per_epoch = 0
# 		for row in reader:
# 		    if count < 1:
# 		        count += 1
# 		        continue
# 		    epoch_count = int(row[0])
# 		    layers[layer_count].append(float(row[1])/1000)
# 		    if epoch_count == epochs - 1:
# 		    	layer_count += 1
# 		    count += 1

fig, axs= plt.subplots(1,1)
axs.set_xlabel('Epoch #', size=15)
if ff_type == 'attention':
	axs.plot(layers[len(layers)-1], label ='Attn. Layer')
	for l in range(len(layers)-1):
		axs.plot(layers[l], label ='FC Layer '+str(l+1))
else:
	for l in range(len(layers)):
		axs.plot(layers[l], label ='BackProp')
axs.tick_params(axis='x', which='minor', bottom=False)
plt.minorticks_on() 
plt.grid(which='major', axis='both', linewidth=1)
plt.grid(which='minor', axis='both',linestyle=':', linewidth=0.6)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend(fontsize=12)
if dataset == "MNIST":
	axs.set_ylabel('Epoch Time\n(sec)', size=15)
else:
	axs.yaxis.set_major_formatter(plt.NullFormatter())
axs.set_ylim(0,50)
axs.set_xlim(0,epochs)
fig.set_size_inches(3, 5)
plt.savefig(os.path.join(plot_directory, dataset, "epoch_time.png"), format='png', bbox_inches='tight', dpi=600)