# generate_param_list.py
import itertools

batch_sizes = [8, 16, 32]
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
epochs_list = [50, 100, 200, 300]

with open("param_list.txt", "w") as f:
    for bs, lr, ep in itertools.product(batch_sizes, learning_rates, epochs_list):
        f.write(f"{bs},{lr},{ep}\n")
