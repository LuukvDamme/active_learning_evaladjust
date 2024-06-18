import numpy as np
import pandas as pd
from statistics import mean
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score
import matplotlib.pyplot as plt

# Constants
TEST_SIZE = 0.25  # The size of the test set as a fraction of the dataset
UNCERTAINTY_RANGE = (0.47, 0.53)  # The range of uncertainty for active learning
ACTIVE_LEARNING_ROUNDS = 10  # The number of rounds of active learning
N_SPLITS = 10  # Number of splits for cross-validation
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
TRAIN_SIZE_RANGE = np.arange(0.01, 0.61, 0.1)
NUM_EXPERIMENTS = 100

# Function to split the dataset into train, test, and unlabelled sets
def split_dataset(dataset, train_size, test_size):
    x = dataset[:, :-1]  # Features
    y = dataset[:, -1]  # Labels
    x_train, x_pool, y_train, y_pool = train_test_split(x, y, train_size=train_size)
    unlabel, x_test, label, y_test = train_test_split(x_pool, y_pool, test_size=test_size)
    return x_train, y_train, x_test, y_test, unlabel, label

# Function to train the model
def train_model(x_train, y_train):
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    return classifier

# Function to perform active learning
def active_learning(x_train, y_train, unlabel, label):
    for _ in range(ACTIVE_LEARNING_ROUNDS):
        classifier = train_model(x_train, y_train)
        y_probab = classifier.predict_proba(unlabel)[:, 0]
        uncertainty_indices = np.where((y_probab >= UNCERTAINTY_RANGE[0]) & (y_probab <= UNCERTAINTY_RANGE[1]))[0]
        x_train = np.append(unlabel[uncertainty_indices, :], x_train, axis=0)
        y_train = np.append(label[uncertainty_indices], y_train)
        unlabel = np.delete(unlabel, uncertainty_indices, axis=0)
        label = np.delete(label, uncertainty_indices)
    return x_train, y_train, unlabel, label

# Function to evaluate the model with cross-validation
def evaluate_model_with_cross_validation(x_train, y_train, n_splits=N_SPLITS):
    classifier = LogisticRegression()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1_scores, precision_scores = [], [], []

    for train_index, val_index in skf.split(x_train, y_train):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        classifier.fit(x_train_fold, y_train_fold)
        accuracies.append(classifier.score(x_val_fold, y_val_fold))

        y_pred = classifier.predict(x_val_fold)
        f1_scores.append(f1_score(y_val_fold, y_pred, average='weighted'))
        precision_scores.append(precision_score(y_val_fold, y_pred, average='weighted'))

    return np.mean(accuracies), np.mean(f1_scores), np.mean(precision_scores), classifier

# Function to evaluate the model on the test set
def evaluate_model_on_test_set(classifier, x_test, y_test):
    accuracy = classifier.score(x_test, y_test)
    y_pred = classifier.predict(x_test)
    return accuracy, f1_score(y_test, y_pred, average='weighted'), precision_score(y_test, y_pred, average='weighted')

# Function to plot the results
def plot_results(train_sizes, avg_active_model_f1_plot, avg_random_sampling_f1_plot, active_test_model_f1, random_test_sampling_f1):
    plt.plot(train_sizes, avg_active_model_f1_plot, label="Active Learning train F1")
    plt.plot(train_sizes, avg_random_sampling_f1_plot, label="Random Sampling train F1")
    plt.plot(train_sizes, active_test_model_f1, label="Active Learning test F1")
    plt.plot(train_sizes, random_test_sampling_f1, label="Random Sampling test F1")

    plt.xlabel("Train Size")
    plt.ylabel("Performance")
    plt.title("Performance vs Train Size")
    plt.legend()
    plt.show()

# Main function
def main():
    # Read the dataset
    dataset = pd.read_csv(DATA_URL).values

    # Impute missing data
    imputer = SimpleImputer(missing_values=0, strategy="mean")
    dataset[:, :-1] = imputer.fit_transform(dataset[:, :-1])

    # Feature scaling
    sc = StandardScaler()
    dataset[:, :-1] = sc.fit_transform(dataset[:, :-1])

    # Initialize lists to store the performance
    train_sizes = []
    avg_active_model_accuracies, avg_random_sampling_accuracies = [], []
    avg_active_model_f1_plot, avg_random_sampling_f1_plot = [], []
    active_test_model_accuracies, random_test_sampling_accuracies = [], []
    active_test_model_f1, random_test_sampling_f1 = [], []

    # Run the experiment for each TRAIN_SIZE value
    for i, train_size in enumerate(TRAIN_SIZE_RANGE):
        print(f"Processing TRAIN_SIZE {train_size:.2f} ({i+1}/{len(TRAIN_SIZE_RANGE)})...")
        active_model_accuracies, random_sampling_accuracies = [], []
        active_model_f1, random_model_f1 = [], []
        active_model_test_accuracies, random_model_test_accuracies = [], []
        active_model_test_f1, random_model_test_f1 = [], []

        for _ in range(NUM_EXPERIMENTS):
            # Split the dataset into train, test, and unlabelled sets
            x_train, y_train, x_test, y_test, unlabel, label = split_dataset(dataset, train_size, TEST_SIZE)

            # Train the model with active learning
            x_train, y_train, unlabel, label = active_learning(x_train, y_train, unlabel, label)

            # Evaluate the model with active learning and cross-validation
            active_model_performance = evaluate_model_with_cross_validation(x_train, y_train)
            active_model_accuracies.append(active_model_performance[0])
            active_model_f1.append(active_model_performance[1])
            active_classifier = active_model_performance[3]

            # Train the model without active learning
            train_size_no_active_learning = x_train.shape[0] / dataset.shape[0]
            x_train, y_train, x_test, y_test, unlabel, label = split_dataset(dataset, train_size_no_active_learning, TEST_SIZE)

            # Evaluate the model without active learning
            random_model_performance = evaluate_model_with_cross_validation(x_train, y_train)
            random_sampling_accuracies.append(random_model_performance[0])
            random_model_f1.append(random_model_performance[1])
            random_classifier = random_model_performance[3]

            # Evaluate the active model on the test set
            active_test_set_performance = evaluate_model_on_test_set(active_classifier, x_test, y_test)
            active_model_test_accuracies.append(active_test_set_performance[0])
            active_model_test_f1.append(active_test_set_performance[1])

            # Evaluate the random model on the test set
            random_test_set_performance = evaluate_model_on_test_set(random_classifier, x_test, y_test)
            random_model_test_accuracies.append(random_test_set_performance[0])
            random_model_test_f1.append(random_test_set_performance[1])

        # Calculate the average performance for the current TRAIN_SIZE value
        avg_active_model_accuracies.append(np.mean(active_model_accuracies))
        avg_random_sampling_accuracies.append(np.mean(random_sampling_accuracies))
        avg_active_model_f1_plot.append(np.mean(active_model_f1))
        avg_random_sampling_f1_plot.append(np.mean(random_model_f1))
        active_test_model_accuracies.append(np.mean(active_model_test_accuracies))
        random_test_sampling_accuracies.append(np.mean(random_model_test_accuracies))
        active_test_model_f1.append(np.mean(active_model_test_f1))
        random_test_sampling_f1.append(np.mean(random_model_test_f1))

        train_sizes.append(train_size)

    # Plot the results
    plot_results(train_sizes, avg_active_model_f1_plot, avg_random_sampling_f1_plot, active_test_model_f1, random_test_sampling_f1)

# Run the main function
if __name__ == "__main__":
    main()
