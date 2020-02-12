import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob

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