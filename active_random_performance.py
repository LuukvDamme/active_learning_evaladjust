import numpy as np
import pandas as pd
from statistics import mean
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, log_loss, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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
    accuracies, f1_scores, precision_scores, losses = [], [], [], []

    for train_index, val_index in skf.split(x_train, y_train):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        classifier.fit(x_train_fold, y_train_fold)
        accuracies.append(classifier.score(x_val_fold, y_val_fold))

        y_pred = classifier.predict(x_val_fold)
        f1_scores.append(f1_score(y_val_fold, y_pred, average='weighted'))
        precision_scores.append(precision_score(y_val_fold, y_pred, average='weighted'))

        y_prob = classifier.predict_proba(x_val_fold)
        losses.append(log_loss(y_val_fold, y_prob))

    return np.mean(accuracies), np.mean(f1_scores), np.mean(precision_scores), np.mean(losses), classifier

# Function to evaluate the model on the test set
def evaluate_model_on_test_set(classifier, x_test, y_test):
    accuracy = classifier.score(x_test, y_test)
    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)
    return accuracy, f1_score(y_test, y_pred, average='weighted'), precision_score(y_test, y_pred, average='weighted'), log_loss(y_test, y_prob), y_prob

# Function to plot F1 scores
def plot_f1_scores(train_sizes, avg_active_model_f1_plot, avg_random_sampling_f1_plot, active_test_model_f1, random_test_sampling_f1):
    plt.figure()
    plt.plot(train_sizes, avg_active_model_f1_plot, label="Active Learning train F1")
    plt.plot(train_sizes, avg_random_sampling_f1_plot, label="Random Sampling train F1")
    plt.plot(train_sizes, active_test_model_f1, label="Active Learning test F1")
    plt.plot(train_sizes, random_test_sampling_f1, label="Random Sampling test F1")
    plt.xlabel("Train Size")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Train Size")
    plt.legend()
    plt.show()

# Function to plot losses
def plot_losses(train_sizes, active_model_losses, random_model_losses):
    plt.figure()
    plt.plot(train_sizes, active_model_losses, label="Active Learning train Loss")
    plt.plot(train_sizes, random_model_losses, label="Random Sampling train Loss")
    plt.xlabel("Train Size")
    plt.ylabel("Log Loss")
    plt.title("Log Loss vs Train Size")
    plt.legend()
    plt.show()

# Function to plot ROC curves
def plot_roc_curve(fpr_active, tpr_active, roc_auc_active, fpr_random, tpr_random, roc_auc_random):
    plt.figure()
    plt.plot(fpr_active, tpr_active, color='darkorange', lw=2, label=f'Active Learning ROC curve (area = {roc_auc_active:.2f})')
    plt.plot(fpr_random, tpr_random, color='blue', lw=2, label=f'Random Sampling ROC curve (area = {roc_auc_random:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def fit_and_plot_gmm_density(x_test, y_test, n_components=2):
    # Fit a Gaussian Mixture Model to the test data
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(x_test)
    
    # Predict the density for each point in the test set
    densities = gmm.score_samples(x_test)
    
    # Plotting the density
    plt.figure(figsize=(10, 6))
    plt.hist(densities, bins=30, density=True, alpha=0.6, color='g')
    plt.title("Density Estimation with Gaussian Mixture Model")
    plt.xlabel("Log Likelihood")
    plt.ylabel("Density")
    plt.show()

    # Visualize the distribution of densities for each class
    plt.figure(figsize=(10, 6))
    for label in np.unique(y_test):
        label_densities = densities[y_test == label]
        plt.hist(label_densities, bins=30, density=True, alpha=0.6, label=f'Class {label}')
    plt.title("Density Estimation by Class with Gaussian Mixture Model")
    plt.xlabel("Log Likelihood")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


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
    active_model_losses, random_model_losses = [], []
    fpr_active, tpr_active, roc_auc_active = [], [], []
    fpr_random, tpr_random, roc_auc_random = [], [], []

    # Run the experiment for each TRAIN_SIZE value
    for i, train_size in enumerate(TRAIN_SIZE_RANGE):
        print(f"Processing TRAIN_SIZE {train_size:.2f} ({i+1}/{len(TRAIN_SIZE_RANGE)})...")
        active_model_accuracies, random_sampling_accuracies = [], []
        active_model_f1, random_model_f1 = [], []
        active_model_test_accuracies, random_model_test_accuracies = [], []
        active_model_test_f1, random_model_test_f1 = [], []
        active_model_losses_temp, random_model_losses_temp = [], []

        for _ in range(NUM_EXPERIMENTS):
            # Split the dataset into train, test, and unlabelled sets
            x_train, y_train, x_test, y_test, unlabel, label = split_dataset(dataset, train_size, TEST_SIZE)

            # Train the model with active learning
            x_train, y_train, unlabel, label = active_learning(x_train, y_train, unlabel, label)

            # Evaluate the model with active learning and cross-validation
            active_model_performance = evaluate_model_with_cross_validation(x_train, y_train)
            active_model_accuracies.append(active_model_performance[0])
            active_model_f1.append(active_model_performance[1])
            active_model_losses_temp.append(active_model_performance[3])
            active_classifier = active_model_performance[4]

            # Train the model without active learning
            train_size_no_active_learning = x_train.shape[0] / dataset.shape[0]
            x_train, y_train, x_test, y_test, unlabel, label = split_dataset(dataset, train_size_no_active_learning, TEST_SIZE)

            # Evaluate the model without active learning
            random_model_performance = evaluate_model_with_cross_validation(x_train, y_train)
            random_sampling_accuracies.append(random_model_performance[0])
            random_model_f1.append(random_model_performance[1])
            random_model_losses_temp.append(random_model_performance[3])
            random_classifier = random_model_performance[4]

            # Evaluate the active model on the test set
            active_test_set_performance = evaluate_model_on_test_set(active_classifier, x_test, y_test)
            active_model_test_accuracies.append(active_test_set_performance[0])
            active_model_test_f1.append(active_test_set_performance[1])

            # Evaluate the random model on the test set
            random_test_set_performance = evaluate_model_on_test_set(random_classifier, x_test, y_test)
            random_model_test_accuracies.append(random_test_set_performance[0])
            random_model_test_f1.append(random_test_set_performance[1])

            # Collect ROC curve data for the last iteration
            fpr_active_temp, tpr_active_temp, _ = roc_curve(y_test, active_test_set_performance[4][:, 1])
            roc_auc_active_temp = auc(fpr_active_temp, tpr_active_temp)
            fpr_random_temp, tpr_random_temp, _ = roc_curve(y_test, random_test_set_performance[4][:, 1])
            roc_auc_random_temp = auc(fpr_random_temp, tpr_random_temp)

        # Calculate the average performance for the current TRAIN_SIZE value
        avg_active_model_accuracies.append(np.mean(active_model_accuracies))
        avg_random_sampling_accuracies.append(np.mean(random_sampling_accuracies))
        avg_active_model_f1_plot.append(np.mean(active_model_f1))
        avg_random_sampling_f1_plot.append(np.mean(random_model_f1))
        active_test_model_accuracies.append(np.mean(active_model_test_accuracies))
        random_test_sampling_accuracies.append(np.mean(random_model_test_accuracies))
        active_test_model_f1.append(np.mean(active_model_test_f1))
        random_test_sampling_f1.append(np.mean(random_model_test_f1))
        active_model_losses.append(np.mean(active_model_losses_temp))
        random_model_losses.append(np.mean(random_model_losses_temp))

        # Collect the ROC curve data
        fpr_active.append(fpr_active_temp)
        tpr_active.append(tpr_active_temp)
        roc_auc_active.append(roc_auc_active_temp)
        fpr_random.append(fpr_random_temp)
        tpr_random.append(tpr_random_temp)
        roc_auc_random.append(roc_auc_random_temp)

    # Plot the results
    plot_f1_scores(TRAIN_SIZE_RANGE, avg_active_model_f1_plot, avg_random_sampling_f1_plot, 
                   active_test_model_f1, random_test_sampling_f1)
    plot_losses(TRAIN_SIZE_RANGE, active_model_losses, random_model_losses)
    plot_roc_curve(fpr_active[-1], tpr_active[-1], roc_auc_active[-1], fpr_random[-1], tpr_random[-1], roc_auc_random[-1])

    # Print the shape of x_test and y_test again after the second split
    print(f"x_test shape (no active learning): {x_test.shape}")
    print(f"y_test shape (no active learning): {y_test.shape}")
    # Fit and plot GMM density for the last test set (as an example)
    fit_and_plot_gmm_density(x_test, y_test)

# Run the main function
if __name__ == "__main__":
    main()