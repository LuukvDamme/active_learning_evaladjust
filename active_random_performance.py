import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, log_loss, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# Constants
TEST_SIZE = 0.25  # The size of the test set as a fraction of the pool set
ACTIVE_LEARNING_ROUNDS = 10  # The number of rounds of active learning
N_SPLITS = 10  # Number of splits for cross-validation
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
TRAIN_SIZE_RANGE = np.arange(0.02, 0.75, 0.01)
NUM_EXPERIMENTS = 1
INITIAL_TRAIN_SIZE_PERCENTAGE = 0.01  # Initial training set size as a fraction of the dataset

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
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x_train, y_train)
    return classifier

# Function to perform active learning
def active_learning(x_train, y_train, unlabel, label, target_train_size, total_data_size):
    rounds = 0
    while len(y_train) < target_train_size and rounds < ACTIVE_LEARNING_ROUNDS:
        classifier = train_model(x_train, y_train)

        if len(unlabel) == 0:
            print("No more samples to select from. Stopping active learning.")
            break
        
        y_probab = classifier.predict_proba(unlabel)[:, 1]  # Probability of class 1
        uncertainties = np.abs(y_probab - 0.5)  # Uncertainty is highest when probability is close to 0.5
        uncertain_indices = np.argsort(uncertainties)[:min(len(uncertainties), target_train_size - len(y_train))]  # Select most uncertain samples

        if len(uncertain_indices) == 0:
            break

        x_train = np.append(unlabel[uncertain_indices, :], x_train, axis=0)
        y_train = np.append(label[uncertain_indices], y_train)
        unlabel = np.delete(unlabel, uncertain_indices, axis=0)
        label = np.delete(label, uncertain_indices)
        rounds += 1
        current_train_percentage = (len(y_train) / total_data_size) * 100
        print(f"Round {rounds}: Added {len(uncertain_indices)} samples, total train size: {len(y_train)}, data used: {current_train_percentage:.2f}%")

    return x_train, y_train, unlabel, label

# Function to evaluate the model with cross-validation
def evaluate_model_with_cross_validation(x_train, y_train, n_splits=N_SPLITS):
    classifier = LogisticRegression(max_iter=1000)
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

# Function to evaluate the model on the test set with weighted F1 score
def evaluate_model_on_test_set_weighted(classifier, x_test, y_test, weights):
    y_pred = classifier.predict(x_test)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', sample_weight=weights)
    accuracy = classifier.score(x_test, y_test)
    y_prob = classifier.predict_proba(x_test)
    return accuracy, f1_weighted, precision_score(y_test, y_pred, average='weighted'), log_loss(y_test, y_prob), y_prob

# Function to plot F1 scores
def plot_f1_scores(train_sizes, avg_active_model_f1_plot, avg_random_sampling_f1_plot, active_test_model_f1, random_test_sampling_f1, weighted_active_f1, weighted_random_f1):
    plt.figure()

    train_sizes_percent = np.array(train_sizes) * 100

    # Define a set of distinct colors
    colors = {
        "Active Learning CV": "#1f77b4",       # Blue
        "Random Sampling CV": "#8c564b",  # Brown
        "Active Learning test": "#2ca02c",     # Green
        "Random Sampling test": "#d62728",     # Red
        "Weighted Active Learning": "#9467bd", # Purple
        "Weighted Random Sampling": "#ff7f0e"       # Orange
    }

    plt.plot(train_sizes_percent, avg_active_model_f1_plot, label="Active Learning CV", color=colors["Active Learning CV"])
    plt.plot(train_sizes_percent, avg_random_sampling_f1_plot, label="Random Sampling CV", color=colors["Random Sampling CV"])
    plt.plot(train_sizes_percent, active_test_model_f1, label="Active Learning test", color=colors["Active Learning test"])
    plt.plot(train_sizes_percent, random_test_sampling_f1, label="Random Sampling test", color=colors["Random Sampling test"])
    plt.plot(train_sizes_percent, weighted_active_f1, label="Weighted Active Learning", color=colors["Weighted Active Learning"], linestyle='--')
    plt.plot(train_sizes_percent, weighted_random_f1, label="Weighted Random Sampling", color=colors["Weighted Random Sampling"], linestyle='--')
    
    plt.xlabel("Train Size (%)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Train Size")
    plt.legend()
    plt.show()

# Function to plot Log Loss
def plot_losses(train_sizes, active_model_losses, random_model_losses):
    train_sizes_percent = [size * 100 for size in train_sizes]

    plt.figure()

    # Define a set of distinct colors
    colors = {
        "Active Learning train Loss": "#1f77b4",  # Blue
        "Random Sampling train Loss": "#ff7f0e"   # Orange
    }

    plt.plot(train_sizes_percent, active_model_losses, label="Active Learning train Loss", color=colors["Active Learning train Loss"])
    plt.plot(train_sizes_percent, random_model_losses, label="Random Sampling train Loss", color=colors["Random Sampling train Loss"])
    
    plt.xlabel("Train Size (%)")
    plt.ylabel("Log Loss")
    plt.title("Log Loss vs Train Size")
    plt.legend()
    plt.show()

# Function to plot ROC Curve
def plot_roc_curve(fpr_active, tpr_active, roc_auc_active, fpr_random, tpr_random, roc_auc_random):
    plt.figure()

    # Define a set of distinct colors
    colors = {
        "Active Learning ROC curve": "#1f77b4",   # Blue
        "Random Sampling ROC curve": "#ff7f0e"    # Orange
    }

    plt.plot(fpr_active, tpr_active, color=colors["Active Learning ROC curve"], lw=2, label=f'Active Learning ROC curve (area = {roc_auc_active:.2f})')
    plt.plot(fpr_random, tpr_random, color=colors["Random Sampling ROC curve"], lw=2, label=f'Random Sampling ROC curve (area = {roc_auc_random:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Function to fit Gaussian Mixture Model and calculate weights
def fit_and_plot_gmm_density(x_train, y_train, x_test, y_test, n_components=2):
    # Fit a Gaussian Mixture Model to the train data
    gmm_train = GaussianMixture(n_components=n_components, random_state=42)
    gmm_train.fit(x_train)
    
    # Predict the density for each point in the train set
    densities_train = gmm_train.score_samples(x_train)
    
    # Fit a Gaussian Mixture Model to the test data
    gmm_test = GaussianMixture(n_components=n_components, random_state=42)
    gmm_test.fit(x_test)
    
    # Predict the density for each point in the test set
    densities_test = gmm_test.score_samples(x_test)
    
    # Calculate weights using densities (importance sampling)
    weights = np.exp(densities_test - densities_train.mean())  # Adjust as necessary for your specific case
    
    return weights, gmm_train, gmm_test

# Function to plot Gaussian Mixture Model ellipsoids
def plot_gmm_ellipsoids(gmm, X, ax, label):
    colors = ['red', 'blue']  # Adjust the colors based on the number of components
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        
        ell = Ellipse(xy=mean, width=v[0], height=v[1],
                      angle=np.degrees(np.arctan2(u[1], u[0])),
                      color=colors[i], alpha=0.5, label=f'{label} Component {i+1}')
        ax.add_patch(ell)
    
    ax.scatter(X[:, 0], X[:, 1], s=10, color='black', label='Data')
    ax.set_title(f'GMM Ellipsoids - {label}')
    ax.legend()

def main():
    # Load and preprocess the dataset
    df = pd.read_csv(DATA_URL, header=None)
    dataset = df.values

    # Impute missing values with the mean of each column
    imputer = SimpleImputer(strategy="mean")
    dataset = imputer.fit_transform(dataset)

    # Standardize the dataset
    scaler = StandardScaler()
    dataset[:, :-1] = scaler.fit_transform(dataset[:, :-1])

    print(dataset.shape)
    # Split the dataset into unlabel, label, x_test, and y_test sets
    x_pool, y_pool = dataset[:, :-1], dataset[:, -1]
    unlabel, label, x_test, y_test = split_pool_set(x_pool, y_pool, TEST_SIZE)

    results = {
        'train_sizes': [],
        'active_test_model_f1': [],
        'random_test_sampling_f1': [],
        'avg_active_model_f1_plot': [],
        'avg_random_sampling_f1_plot': [],
        'active_model_losses': [],
        'random_model_losses': [],
        'weighted_active_f1': [],
        'weighted_random_f1': []
    }

    initial_train_size = int(len(dataset) * INITIAL_TRAIN_SIZE_PERCENTAGE)
    total_data_size = len(dataset)

    for target_train_size_fraction in TRAIN_SIZE_RANGE:
        target_train_size = int(len(dataset) * target_train_size_fraction)
        results['train_sizes'].append(target_train_size_fraction)

        active_f1_scores = []
        random_f1_scores = []
        active_losses = []
        random_losses = []
        active_test_f1_scores = []
        random_test_f1_scores = []
        weighted_active_f1_scores = []
        weighted_random_f1_scores = []

        for experiment in range(NUM_EXPERIMENTS):
            # Split the dataset into the initial training and pool sets
            x_train, y_train, x_pool, y_pool = split_dataset_initial(dataset, initial_train_size)
            current_unlabel, current_label = unlabel.copy(), label.copy()

            # Active Learning
            x_train_active, y_train_active, _, _ = active_learning(x_train, y_train, current_unlabel, current_label, target_train_size, total_data_size)
            avg_accuracy_active, avg_f1_active, avg_precision_active, avg_loss_active, classifier_active = evaluate_model_with_cross_validation(x_train_active, y_train_active)
            active_f1_scores.append(avg_f1_active)
            active_losses.append(avg_loss_active)
            accuracy_active, f1_active, _, loss_active, y_prob_active = evaluate_model_on_test_set(classifier_active, x_test, y_test)
            active_test_f1_scores.append(f1_active)

            # Random Sampling
            if len(y_pool) > 0:
                random_indices = np.random.choice(len(y_pool), size=target_train_size - initial_train_size, replace=False)
                x_train_random = np.append(x_train, x_pool[random_indices], axis=0)
                y_train_random = np.append(y_train, y_pool[random_indices])
            else:
                x_train_random, y_train_random = x_train, y_train

            avg_accuracy_random, avg_f1_random, avg_precision_random, avg_loss_random, classifier_random = evaluate_model_with_cross_validation(x_train_random, y_train_random)
            random_f1_scores.append(avg_f1_random)
            random_losses.append(avg_loss_random)
            accuracy_random, f1_random, _, loss_random, y_prob_random = evaluate_model_on_test_set(classifier_random, x_test, y_test)
            random_test_f1_scores.append(f1_random)

            # Fit Gaussian Mixture Model to calculate weights
            weights, gmm_train_active, gmm_test_active = fit_and_plot_gmm_density(x_train_active, y_train_active, x_test, y_test)

            # Evaluate models with weighted F1 score on test set
            _, f1_active_weighted, _, _, _ = evaluate_model_on_test_set_weighted(classifier_active, x_test, y_test, weights)
            _, f1_random_weighted, _, _, _ = evaluate_model_on_test_set_weighted(classifier_random, x_test, y_test, weights)

            weighted_active_f1_scores.append(f1_active_weighted)
            weighted_random_f1_scores.append(f1_random_weighted)

        results['avg_active_model_f1_plot'].append(np.mean(active_f1_scores))
        results['avg_random_sampling_f1_plot'].append(np.mean(random_f1_scores))
        results['active_model_losses'].append(np.mean(active_losses))
        results['random_model_losses'].append(np.mean(random_losses))
        results['active_test_model_f1'].append(np.mean(active_test_f1_scores))
        results['random_test_sampling_f1'].append(np.mean(random_test_f1_scores))
        results['weighted_active_f1'].append(np.mean(weighted_active_f1_scores))
        results['weighted_random_f1'].append(np.mean(weighted_random_f1_scores))

    # Plot results
    plot_f1_scores(results['train_sizes'], results['avg_active_model_f1_plot'], results['avg_random_sampling_f1_plot'], results['active_test_model_f1'], results['random_test_sampling_f1'], results['weighted_active_f1'], results['weighted_random_f1'])
    plot_losses(results['train_sizes'], results['active_model_losses'], results['random_model_losses'])

    # Compute ROC curves
    fpr_active, tpr_active, _ = roc_curve(y_test, y_prob_active[:, 1])
    roc_auc_active = auc(fpr_active, tpr_active)

    fpr_random, tpr_random, _ = roc_curve(y_test, y_prob_random[:, 1])
    roc_auc_random = auc(fpr_random, tpr_random)

    # Plot ROC curves
    plot_roc_curve(fpr_active, tpr_active, roc_auc_active, fpr_random, tpr_random, roc_auc_random)

    # Plot GMM Ellipsoids
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_gmm_ellipsoids(gmm_train_active, x_train_active, ax[0], 'Active Learning')
    plot_gmm_ellipsoids(gmm_test_active, x_test, ax[1], 'Test Set')
    plt.show()

if __name__ == "__main__":
    main()
