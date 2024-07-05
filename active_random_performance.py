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
TEST_SIZE = 0.25  # The size of the test set as a fraction of the pool set
UNCERTAINTY_RANGE = (0.47, 0.53)  # The range of uncertainty for active learning
ACTIVE_LEARNING_ROUNDS = 10  # The number of rounds of active learning
N_SPLITS = 10  # Number of splits for cross-validation
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
TRAIN_SIZE_RANGE = np.arange(0.01, 0.31, 0.1)
NUM_EXPERIMENTS = 1

# Function to split the dataset into train, pool, test, and unlabelled sets
def split_dataset_initial(dataset, train_size):
    x = dataset[:, :-1]  # Features
    y = dataset[:, -1].astype(int)  # Labels as integers
    x_train, x_pool, y_train, y_pool = train_test_split(x, y, train_size=train_size, random_state=42, stratify=y)
    return x_train, y_train, x_pool, y_pool

def split_pool_set(x_pool, y_pool, test_size):
    unlabel, x_test, label, y_test = train_test_split(x_pool, y_pool, test_size=test_size, random_state=42, stratify=y_pool)
    return unlabel, label, x_test, y_test

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

def fit_and_plot_gmm_density(x_train, y_train, x_test, y_test, n_components=2):
    # Fit a Gaussian Mixture Model to the train data
    gmm_train = GaussianMixture(n_components=n_components, random_state=42)
    gmm_train.fit(x_train)
    
    # Predict the density for each point in the train set
    densities_train = gmm_train.score_samples(x_train)
    
    # Plotting the density for the train set
    plt.figure(figsize=(10, 6))
    plt.hist(densities_train, bins=30, density=True, alpha=0.6, color='b', label='Train')
    plt.title("Density Estimation with Gaussian Mixture Model (Train Set)")
    plt.xlabel("Log Likelihood")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # Fit a Gaussian Mixture Model to the test data
    gmm_test = GaussianMixture(n_components=n_components, random_state=42)
    gmm_test.fit(x_test)
    
    # Predict the density for each point in the test set
    densities_test = gmm_test.score_samples(x_test)
    
    # Plotting the density for the test set
    plt.figure(figsize=(10, 6))
    plt.hist(densities_test, bins=30, density=True, alpha=0.6, color='g', label='Unlabeled')
    plt.title("Density Estimation with Gaussian Mixture Model (Unlabeled Set)")
    plt.xlabel("Log Likelihood")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    
    # Combined plot for train and test set densities
    plt.figure(figsize=(10, 6))
    plt.hist(densities_train, bins=30, density=True, alpha=0.6, color='b', label='Train')
    plt.hist(densities_test, bins=30, density=True, alpha=0.6, color='g', label='Unlabeled')
    plt.title("Density Estimation with Gaussian Mixture Model (Train and Unlabeled Sets)")
    plt.xlabel("Log Likelihood")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # Visualize the distribution of densities for each class in the train set
    plt.figure(figsize=(10, 6))
    for label in np.unique(y_train):
        densities_label = gmm_train.score_samples(x_train[y_train == label])
        plt.hist(densities_label, bins=30, density=True, alpha=0.6, label=f'Train - Class {label}')
    
    plt.title("Density Estimation for Each Class in the Train Set")
    plt.xlabel("Log Likelihood")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# Main function to run the experiments
def main():
    # Load and preprocess the dataset
    dataset = pd.read_csv(DATA_URL).to_numpy()
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    dataset = imputer.fit_transform(dataset)
    dataset = scaler.fit_transform(dataset)

    # Ensure labels are integers
    dataset[:, -1] = dataset[:, -1].astype(int)

    # Split the pool set into unlabelled and test sets (static split)
    x_pool, y_pool = dataset[:, :-1], dataset[:, -1]
    unlabel, label, x_test, y_test = split_pool_set(x_pool, y_pool, TEST_SIZE)

    train_sizes = []
    active_test_model_f1, active_test_model_precision, active_test_model_accuracy = [], [], []
    random_test_sampling_f1, random_test_sampling_precision, random_test_sampling_accuracy = [], [], []
    avg_active_model_f1_plot, avg_random_sampling_f1_plot = [], []
    active_model_losses, random_model_losses = [], []

    for train_size in TRAIN_SIZE_RANGE:
        for experiment in range(NUM_EXPERIMENTS):
            x_train, y_train, x_pool, y_pool = split_dataset_initial(dataset, train_size)
            
            # Ensure test and unlabelled sets are the same for each iteration
            current_unlabel, current_label = unlabel.copy(), label.copy()

            # Active Learning
            x_train_active, y_train_active, _, _ = active_learning(x_train, y_train, current_unlabel, current_label)
            avg_accuracy_active, avg_f1_active, avg_precision_active, avg_loss_active, classifier_active = evaluate_model_with_cross_validation(x_train_active, y_train_active)
            active_model_losses.append(avg_loss_active)
            avg_active_model_f1_plot.append(avg_f1_active)

            accuracy_active, f1_active, precision_active, loss_active, y_prob_active = evaluate_model_on_test_set(classifier_active, x_test, y_test)
            active_test_model_f1.append(f1_active)
            active_test_model_precision.append(precision_active)
            active_test_model_accuracy.append(accuracy_active)

            # Random Sampling
            avg_accuracy_random, avg_f1_random, avg_precision_random, avg_loss_random, classifier_random = evaluate_model_with_cross_validation(x_train, y_train)
            random_model_losses.append(avg_loss_random)
            avg_random_sampling_f1_plot.append(avg_f1_random)

            accuracy_random, f1_random, precision_random, loss_random, y_prob_random = evaluate_model_on_test_set(classifier_random, x_test, y_test)
            random_test_sampling_f1.append(f1_random)
            random_test_sampling_precision.append(precision_random)
            random_test_sampling_accuracy.append(accuracy_random)
        
        train_sizes.append(train_size)

    # Plot results
    plot_f1_scores(train_sizes, avg_active_model_f1_plot, avg_random_sampling_f1_plot, active_test_model_f1, random_test_sampling_f1)
    plot_losses(train_sizes, active_model_losses, random_model_losses)
    
    # Plot ROC curve for the final model
    fpr_active, tpr_active, _ = roc_curve(y_test, y_prob_active[:, 1])
    roc_auc_active = auc(fpr_active, tpr_active)
    fpr_random, tpr_random, _ = roc_curve(y_test, y_prob_random[:, 1])
    roc_auc_random = auc(fpr_random, tpr_random)
    plot_roc_curve(fpr_active, tpr_active, roc_auc_active, fpr_random, tpr_random, roc_auc_random)

    # Fit and plot GMM densities
    fit_and_plot_gmm_density(x_train_active, y_train_active, unlabel, label)

if __name__ == "__main__":
    main()
