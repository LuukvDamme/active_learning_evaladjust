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

# Function to split the dataset into train, test, and unlabelled sets
def split_dataset(dataset, train_size, test_size):
    x = dataset[:, :-1]  # Features
    y = dataset[:, -1]  # Labels
    x_train, x_pool, y_train, y_pool = train_test_split(
        x, y, train_size=train_size)
    unlabel, x_test, label, y_test = train_test_split(
        x_pool, y_pool, test_size=test_size)
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
def evaluate_model_with_cross_validation(x_train, y_train, n_splits=5):
    classifier = LogisticRegression()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
    precision_scores = []
    best_model = None
    best_score = 0

    for train_index, val_index in skf.split(x_train, y_train):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        classifier.fit(x_train_fold, y_train_fold)
        accuracy = classifier.score(x_val_fold, y_val_fold)
        accuracies.append(accuracy)

        y_pred = classifier.predict(x_val_fold)
        f1_score_val = f1_score(y_val_fold, y_pred, average='weighted')
        f1_scores.append(f1_score_val)

        precision_score_val = precision_score(y_val_fold, y_pred, average='weighted')
        precision_scores.append(precision_score_val)

        if accuracy > best_score:
            best_score = accuracy
            best_model = classifier

    return np.mean(accuracies), np.mean(f1_scores), np.mean(precision_scores), best_model

# Function to evaluate the model on the test set
def evaluate_model_on_test_set(classifier, x_test, y_test):
    accuracy = classifier.score(x_test, y_test)
    y_pred = classifier.predict(x_test)
    f1_score_val = f1_score(y_test, y_pred, average='weighted')
    precision_score_val = precision_score(y_test, y_pred, average='weighted')
    return accuracy, f1_score_val, precision_score_val

# Main function
def main():
    # Read the dataset
    dataset = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data").values[:, ]

    # Impute missing data
    imputer = SimpleImputer(missing_values=0, strategy="mean")
    imputer = imputer.fit(dataset[:, :-1])
    dataset[:, :-1] = imputer.transform(dataset[:, :-1])

    # Feature scaling
    sc = StandardScaler()
    dataset[:, :-1] = sc.fit_transform(dataset[:, :-1])

    # Initialize lists to store the performance
    active_model_accuracies = []
    random_sampling_accuracies = []
    avg_active_model_accuracies = []
    avg_random_sampling_accuracies = []
    active_model_f1 = []
    random_model_f1 = []
    avg_active_model_f1_plot = []
    avg_random_sampling_f1_plot = []
    train_sizes = []
    active_model_test_accuracies = []
    random_model_test_accuracies = []
    active_test_model_accuracies = []
    random_test_sampling_accuracies = []
    active_model_test_f1 = []
    random_model_test_f1 = []
    active_test_model_f1 = []
    random_test_sampling_f1 = []

    # Run the experiment 100 times for each TRAIN_SIZE value
    for train_size in np.arange(0.01, 0.95, 0.01):
        for _ in range(100):
            # Split the dataset into train, test, and unlabelled sets
            x_train, y_train, x_test, y_test, unlabel, label = split_dataset(
                dataset, train_size, TEST_SIZE)

            # Train the model with active learning
            x_train, y_train, unlabel, label = active_learning(
                x_train, y_train, unlabel, label)

            # Evaluate the model with active learning and cross-validation
            active_model_performance = evaluate_model_with_cross_validation(x_train, y_train)
            active_model_accuracy = active_model_performance[0]
            active_model_accuracies.append(active_model_accuracy)
            active_model_f1_loop = active_model_performance[1]
            active_model_f1.append(active_model_f1_loop)
            active_classifier = active_model_performance[3]

            # Train the model without active learning
            train_size_no_active_learning = x_train.shape[0] / dataset.shape[0]
            x_train, y_train, x_test, y_test, unlabel, label = split_dataset(
                dataset, train_size_no_active_learning, TEST_SIZE)

            # Evaluate the model without active learning
            random_model_performance = evaluate_model_with_cross_validation(x_train, y_train)
            random_sampling_accuracy = random_model_performance[0]
            random_sampling_accuracies.append(random_sampling_accuracy)
            random_model_f1_loop = random_model_performance[1]
            random_model_f1.append(random_model_f1_loop)
            random_classifier = random_model_performance[3]


            # Evaluate the active model on the test set
            test_set_performance = evaluate_model_on_test_set(active_classifier, x_test, y_test)
            test_set_accuracy = test_set_performance[0]
            test_set_f1 = test_set_performance[1]
            test_set_precision = test_set_performance[2]
            active_model_test_accuracies.append(test_set_accuracy)
            active_model_test_f1.append(test_set_f1)


            # Evaluate the random model on the test set
            test_set_performance = evaluate_model_on_test_set(random_classifier, x_test, y_test)
            test_set_accuracy = test_set_performance[0]
            test_set_f1 = test_set_performance[1]
            test_set_precision = test_set_performance[2]
            random_model_test_accuracies.append(test_set_accuracy)
            random_model_test_f1.append(test_set_f1)


        # Calculate the average accuracies for the current TRAIN_SIZE value
        avg_active_model_accuracy = np.mean(
            active_model_accuracies)
        avg_random_sampling_accuracy = np.mean(
            random_sampling_accuracies)
        avg_active_model_f1 = np.mean(
            active_model_f1)
        avg_random_sampling_f1 = np.mean(
            random_model_f1)
        
        # Calculate the average accuracies for the current TEST_SIZE value
        avg_active_test_model_accuracy = np.mean(
            active_model_test_accuracies)
        avg_random_test_sampling_accuracy = np.mean(
            random_model_test_accuracies)        
        avg_active_test_model_f1 = np.mean(
            active_model_test_f1)
        avg_random_test_sampling_f1 = np.mean(
            random_model_test_f1)
        

        print("Average accuracy with Active Learning for TRAIN_SIZE = {:.2f}: {:.2f}%".format(
            train_size, avg_active_model_accuracy))
        print("Average accuracy with Random Sampling for TRAIN_SIZE = {:.2f}: {:.2f}%".format(
            train_size, avg_random_sampling_accuracy))
        print("Difference for TRAIN_SIZE = {:.2f}: {:.2f}%\n".format(
            train_size, (avg_active_model_accuracy - avg_random_sampling_accuracy)))
        
        print("Average f1 with Active Learning for TRAIN_SIZE = {:.2f}: {:.2f}%".format(
            train_size, avg_active_model_f1))
        print("Average f1 with Random Sampling for TRAIN_SIZE = {:.2f}: {:.2f}%".format(
            train_size, avg_random_sampling_f1))
        print("Difference for TRAIN_SIZE = {:.2f}: {:.2f}%\n".format(
            train_size, (avg_active_model_f1 - avg_random_sampling_f1)))
        
        print("Average acc with Active Learning for TEST_SIZE = {:.2f}: {:.2f}%".format(
            train_size, avg_active_test_model_accuracy))
        print("Average acc with Random Sampling for TEST_SIZE = {:.2f}: {:.2f}%".format(
            train_size, avg_random_test_sampling_accuracy))
        print("Difference for TRAIN_SIZE = {:.2f}: {:.2f}%\n".format(
            train_size, (avg_active_test_model_accuracy - avg_random_test_sampling_accuracy)))
       
        print("Average f1 with Active Learning for TEST_SIZE = {:.2f}: {:.2f}%".format(
            train_size, avg_active_test_model_f1))
        print("Average f1 with Random Sampling for TEST_SIZE = {:.2f}: {:.2f}%".format(
            train_size, avg_random_test_sampling_f1))
        print("Difference for TRAIN_SIZE = {:.2f}: {:.2f}%\n".format(
            train_size, (avg_active_test_model_f1 - avg_random_test_sampling_f1)))

        # Store the average accuracies and train size for plotting
        train_sizes.append(train_size)
        active_model_accuracies.append(avg_active_model_accuracy)
        random_sampling_accuracies.append(avg_random_sampling_accuracy)
        avg_active_model_accuracies.append(avg_active_model_accuracy)
        avg_random_sampling_accuracies.append(avg_random_sampling_accuracy)

        active_model_f1.append(avg_active_model_f1)
        random_model_f1.append(avg_random_sampling_f1)
        avg_active_model_f1_plot.append(avg_active_model_f1)
        avg_random_sampling_f1_plot.append(avg_random_sampling_f1)

        active_test_model_accuracies.append(avg_active_test_model_accuracy)
        random_test_sampling_accuracies.append(avg_random_test_sampling_accuracy)
        active_test_model_f1.append(avg_active_test_model_f1)
        random_test_sampling_f1.append(avg_random_test_sampling_f1)

        # Reset the accuracy lists for the next TRAIN_SIZE value
        active_model_accuracies = []
        random_sampling_accuracies = []
        active_model_f1 = []
        random_model_f1 = []
        active_model_test_accuracies = []
        random_model_test_accuracies = []
        active_model_test_f1 = []
        random_model_test_f1 = []



    # Plot the results
    plt.plot(train_sizes, avg_active_model_accuracies, label="Active Learning train acc")
    plt.plot(train_sizes, avg_random_sampling_accuracies, label="Random Sampling train acc")
    plt.plot(train_sizes, avg_active_model_f1_plot, label="Active Learning train F1")
    plt.plot(train_sizes, avg_random_sampling_f1_plot, label="Random Sampling train F1")

    plt.plot(train_sizes, active_test_model_accuracies, label="Active Learning test acc")
    plt.plot(train_sizes, random_test_sampling_accuracies, label="Random Sampling test acc")
    plt.plot(train_sizes, active_test_model_f1, label="Active Learning test f1")
    plt.plot(train_sizes, random_test_sampling_f1, label="Random Sampling test f1")

    plt.xlabel("Train Size")
    plt.ylabel("Performance")
    plt.title("Performance vs Train Size")
    plt.legend()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
