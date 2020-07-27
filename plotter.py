import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import sys
'''
log_dir = 'relcopylogs/'
#names = ['van100rms0.0002modr1', 'van200rms0.0002modr1', 'van300rms0.0002modr','van500rms0.0002modr', 'van1000rms0.0002modr']
#names = ['100adam0.0002tanh', '200adam0.0002tanh', '300adam0.0002tanh', '500adam0.0002tanh', '1000adam0.0002tanh']
#names = ['lstm100', 'lstm200', 'lstm300', 'lstm500', 'lstm1000']
#names = ['100adam0.001modr', '200rms0.0002modr', '300rms0.0002modr','500rms0.0002modr', '1000rms0.0002modr', '2000rms0.0002modr']
names = ['van100rms0.0002modr', 'van200rms0.0002modr', 'van300rms0.0002modr', 'van500rms0.0002modr', 'van1000rms0.0002modr', 'van2000rms0.0002modr',]
labels = ['T = 100', 'T = 200', 'T = 300', 'T = 500', 'T = 1000', 'T = 2000']
arr = []
save_name = 'default.png'

for i in range(len(names)):
	fname = glob.glob(log_dir + names[i] + '/*')[0]
	na = []
	for event in tf.compat.v1.train.summary_iterator(fname):
		for value in event.summary.value:
			if value.tag == 'Loss':
				na.append(value.simple_value)

	arr.append(np.array(na))

fig, ax = plt.subplots(figsize=(7, 4))
for i in range(len(arr)):
	va = np.arange(arr[i].shape[0])
	if i < 3:
		va = va * 5
	if i == 3:
		va = va * 3
	if i == 4:
		va = va * 4
	plt.plot(va, arr[i], label=labels[i])

plt.title('Copy Task - orth-RNN')
plt.ylabel('Cross-Entropy Loss')
plt.xlabel('Number of Updates')
plt.ylim(0.0, 0.05)
plt.xlim(0, 1000)
plt.legend(loc='upper right')
plt.savefig(save_name)
plt.close()
'''

models = ['RNN', 'LSTM', 'MemRNN', 'SAB', 'RelLSTM']
labels = ['3', '8', '18', '25']
grads = []
for i in range(4):
	'''
	if i == 3:
		f = open('grad_np_files/dLdh_denoise_SAB_1200_norms.txt', 'r')
		data = []
		for val in f:
			val = float(val)
			data.append(val)
		data = np.array(data)
		grads.append(np.flip(np.log(data)))
		continue
	'''
	x = labels[i]
	f = open('grad_np_files/{}rsize.pkl'.format(x), 'rb')
	data = pickle.load(f)
	data = np.flip(np.log(np.array([x.item() for x in data])))
	grads.append(data)
	f.close()
grads = np.array(grads)
colors = ['blue', 'orange', 'green', 'red']

plt.title('Gradient norms plot on the Denoise Task', fontsize=15)
plt.ylabel('log(||dL/dh_t||)', fontsize=15)
plt.xlabel('Number of time steps t', fontsize=15)
plt.plot(grads[0], label=labels[0], color=colors[0])
plt.plot(grads[1], label=labels[1], color=colors[1])
plt.plot(grads[2], label=labels[2], color=colors[2])
plt.plot(grads[3], label=labels[3], color=colors[3])
#plt.plot(grads[4], label=labels[4], color=colors[4])
plt.legend(fontsize=12)
plt.savefig('temp.png')
sys.exit(0)

new_g = np.zeros((5, 4))
for i in range(grads.shape[1]):
	if grads[3][i] == float('-inf'):
		grads[3][i] = 0

for i in range(5):
	for j in range(4):
		arr = grads[i, j*200: (j+1)*200]
		new_g[i][j] = arr.mean()

colors = ['blue', 'orange', 'green', 'red', 'purple']
markers = ['o', 'x', '^']
mems = np.array([[492,498,506,514], [496,506,516,526], [1028,2320,4400,7356], [595,1316,2482,4035] ,[598,702,804,908]]) / 1024.0
labels = ['T = 400', 'T = 600', 'T = 800']
for i in range(5):
	for j in range(1, 4):
		if i == 0:
			plt.scatter(mems[i][j], new_g[i][j], c='black', marker=markers[j-1], label=labels[j-1])
		plt.scatter(mems[i][j], new_g[i][j], c=colors[i], marker=markers[j-1])

x, y = [float('inf')], [float('inf')]
#plt.scatter([1], [1], c='black', marker=markers[0], label='T=400')
plt.title('Gradient Norms vs GPU usage', fontsize=15)
plt.xlabel('Max GPU usage in Gb', fontsize=15)
plt.ylabel('Average log(||dL/dh_t||)', fontsize=15)
plt.legend(fontsize=13)
plt.savefig('scatter.png')