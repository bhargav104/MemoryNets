# Relevancy Screening Mechanism in self-attentive RNNs and LSTMs

This repository contains the code used for the paper [Untangling tradeoffs between recurrence and self-attention in neural networks](https://arxiv.org/abs/2006.09471).

## Software Requirements

Python 3 and Pytorch 1.4.0

## Experiments

To run copy task with a time delay of `1000` steps using RelRNN, execute the following command:

`python copytask.py --log --T=1000 --net-type=RelMemRNN --lr=0.0002 --nonlin=tanh --name=enter_experiment_dir_name_here`

To run the denoise task with a time delay of `1000` steps using RelLSTM, execute the following command:

`python denoisetask.py --log --T=1000 --net-type=RelLSTM --lr=0.001 --name=enter_experiment_dir_name_here`

To run the transfer copy task with a time delay of `5000` steps, execute the following command:

`python transfer.py --T=5000 --net-type=type_of_network --name=name_of_saved_model`

To run the permuted sequential MNIST task using RelMemRNN, execute the following command:

`python sMNIST.py --log --net-type=RelMemRNN --adam --name=enter_experiment_dir_name_here`
and to run it using RelLSTM, use the command:
`python pixelmnist.py --permute=True --algo=RelLSTM --save-dir=enter_experiment_dir_name_here`
