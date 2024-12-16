#from the main directory run python plotting.py --loss-file learner-checkpoints/[MODEL LOSSES NAME].csv
import argparse
import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Plot intermediate losses from a CSV file.")
    parser.add_argument('--loss_file', type=str, required=True, help='Path to the intermediate losses CSV file.')
    return parser.parse_args()

def main():
    args = parse_args()
    csv_file = args.loss_file

    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_filename = 'plots/' + base_name + ".png"

    data = []

    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row['epoch'])
            step = int(row['step'])
            loss = float(row['loss'])
            data.append((epoch, step, loss))
    
    epoch_max_step = defaultdict(int)
    for (epoch, step, loss) in data:
        if step > epoch_max_step[epoch]:
            epoch_max_step[epoch] = step

    epoch_lengths = {}
    for e, max_st in epoch_max_step.items():
        epoch_lengths[e] = max_st + 1

    sorted_epochs = sorted(epoch_lengths.keys())
    epoch_offset = {}
    cumulative = 0
    for e in sorted_epochs:
        epoch_offset[e] = cumulative
        cumulative += epoch_lengths[e]

    total_steps_list = []
    losses = []
    for (epoch, step, loss) in data:
        total_steps = epoch_offset[epoch] + step
        total_steps_list.append(total_steps)
        losses.append(loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(total_steps_list, losses, marker='o', linestyle='-', color='b')
    plt.title('Intermediate Losses Over Total Steps')
    plt.xlabel('Total Steps')
    plt.ylabel('Average Loss')
    plt.grid(True)
    
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    main()
