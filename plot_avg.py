import re
from collections import defaultdict
from statistics import mean
import w2v

# Initialize a dictionary to store metrics
metrics = defaultdict(list)

# Read the file and extract metric values
with open("final_results.txt", "r") as file:
    for line in file:
        # Find the metric name and values
        match = re.match(r"(\w+): \[(.*?)\]", line.strip())

        
        if match:
            metric_name = match.group(1)
            # Convert the values from string to list of floats
            values = list(map(float, match.group(2).split(", ")))
            metrics[metric_name].append(values)

# Calculate the average for each index across each metric
averaged_metrics = {name: [mean(values) for values in zip(*data)] for name, data in metrics.items()}

# Generate train_sizes based on the length of any of the averaged metric lists
train_sizes = [size / 100 for size in range(1, len(next(iter(averaged_metrics.values()))) + 1)]



# Plot using the plot_f1_scores function
w2v.plot_f1_scores(
    train_sizes,
    averaged_metrics['avg_active_f1'], 
    averaged_metrics['random_sampling_f1'], 
    averaged_metrics['active_test_f1'], 
    averaged_metrics['random_test_f1'], 
    averaged_metrics['weighted_active_f1'], 
    averaged_metrics['weighted_random_f1'], 
    averaged_metrics['weighted_train_active_f1'], 
    averaged_metrics['weighted_train_random_f1']
)
