import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, log_loss, roc_curve, auc, precision_recall_curve
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from matplotlib.patches import Ellipse

# Constants
DATA_FILE = 'SMSSpamCollection'
INITIAL_TRAIN_SIZE_PERCENTAGE = 0.01
TRAIN_SIZE_RANGE = np.arange(0.02, 0.74, 0.01)
NUM_EXPERIMENTS = 100
ACTIVE_LEARNING_ROUNDS = 10
N_SPLITS = 10
TEST_SIZE = 0.25  # The size of the test set as a fraction of the pool set

def load_sms_spam_collection(file_path):
    df = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'message'], encoding='utf-8')
    return df

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return tokens

def vectorize_text(texts):
    tokenized_texts = [preprocess_text(text) for text in texts]
    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    vectors = np.array([np.mean([model.wv[word] for word in preprocess_text(text) if word in model.wv] or [np.zeros(model.vector_size)], axis=0) for text in texts])
    return vectors

def preprocess_and_vectorize(file_path):
    df = load_sms_spam_collection(file_path)
    vectors = vectorize_text(df['message'])
    labels = df['label'].apply(lambda x: 1 if x == 'spam' else 0).values
    dataset = np.hstack((vectors, labels.reshape(-1, 1)))
    return dataset

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
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    classifier.fit(x_train, y_train)
    return classifier

# Function to perform active learning
def active_learning(x_train, y_train, unlabel, label, target_train_size, total_data_size):
    rounds = 0
    if len(y_train) < target_train_size:
        print('smaller')
    else:
        print('larger')
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

        # Calculate and print the ratio of each label
        spam_count = np.sum(y_train == 1)
        ham_count = np.sum(y_train == 0)
        spam_ratio = spam_count / len(y_train) * 100
        ham_ratio = ham_count / len(y_train) * 100

        print(f"Round {rounds}: Added {len(uncertain_indices)} samples, total train size: {len(y_train)}, data used: {current_train_percentage:.2f}%")
        print(f"Label ratios - Spam: {spam_ratio:.2f}%, Ham: {ham_ratio:.2f}%")

    return x_train, y_train, unlabel, label


# Function to evaluate the model with cross-validation
def evaluate_model_with_cross_validation(x_train, y_train, n_splits=N_SPLITS):
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1_scores, precision_scores, losses = [], [], [], []

    for train_index, val_index in skf.split(x_train, y_train):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        classifier.fit(x_train_fold, y_train_fold)
        accuracies.append(classifier.score(x_val_fold, y_val_fold))

        y_pred = classifier.predict(x_val_fold)
        f1_scores.append(f1_score(y_val_fold, y_pred, average='weighted', zero_division=1))
        precision_scores.append(precision_score(y_val_fold, y_pred, average='weighted', zero_division=1))

        y_prob = classifier.predict_proba(x_val_fold)
        losses.append(log_loss(y_val_fold, y_prob))

    return np.mean(accuracies), np.mean(f1_scores), np.mean(precision_scores), np.mean(losses), classifier

# Function to evaluate the model on the test set
def evaluate_model_on_test_set(classifier, x_test, y_test):
    accuracy = classifier.score(x_test, y_test)
    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)
    return (
        accuracy,
        f1_score(y_test, y_pred, average='weighted', zero_division=1),
        precision_score(y_test, y_pred, average='weighted', zero_division=1),
        log_loss(y_test, y_prob),
        y_prob,
    )

# Function to evaluate the model on the test set with weighted F1 score
def evaluate_model_on_test_set_weighted(classifier, x_test, y_test, weights):
    y_pred = classifier.predict(x_test)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=1, sample_weight=weights)
    accuracy = classifier.score(x_test, y_test)
    y_prob = classifier.predict_proba(x_test)
    return (
        accuracy,
        f1_weighted,
        precision_score(y_test, y_pred, average='weighted', zero_division=1),
        log_loss(y_test, y_prob),
        y_prob,
    )

# Function to compute GMM-based weights on the test set
def compute_gmm_weights(x_test, y_test):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(x_test)
    probs = gmm.predict_proba(x_test)
    weights = np.zeros_like(y_test, dtype=float)
    weights[y_test == 1] = probs[y_test == 1, 1]  # Weight for spam
    weights[y_test == 0] = probs[y_test == 0, 0]  # Weight for ham
    return weights

# Function to compute GMM-based weights on the training set
def compute_gmm_weights_for_train(x_train, y_train):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(x_train)
    probs = gmm.predict_proba(x_train)
    weights = np.zeros_like(y_train, dtype=float)
    weights[y_train == 1] = probs[y_train == 1, 1]  # Weight for spam
    weights[y_train == 0] = probs[y_train == 0, 0]  # Weight for ham
    return weights

# Function to evaluate the model on the training set with weighted F1 score
def evaluate_model_on_train_set_weighted(classifier, x_train, y_train, weights):
    y_pred = classifier.predict(x_train)
    f1_weighted = f1_score(y_train, y_pred, average='weighted', zero_division=1, sample_weight=weights)
    accuracy = classifier.score(x_train, y_train)
    y_prob = classifier.predict_proba(x_train)
    return (
        accuracy,
        f1_weighted,
        precision_score(y_train, y_pred, average='weighted', zero_division=1),
        log_loss(y_train, y_prob),
        y_prob,
    )

# Function to plot ROC curve
def plot_roc_curve(y_true, y_probs, label):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(y_true, y_probs, label):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(recall, precision, label=f'{label} (PR Curve)')

def plot_f1_scores(train_sizes, avg_active_model_f1_plot, avg_random_sampling_f1_plot, active_test_model_f1, random_test_sampling_f1, weighted_active_f1, weighted_random_f1, weighted_f1_train_active, weighted_f1_train_random):
    plt.figure()

    train_sizes_percent = np.array(train_sizes) * 100

    # Define a set of distinct colors
    colors = {
        "Active Learning CV": "#1f77b4",       # Blue
        "Random Sampling CV": "#8c564b",  # Brown
        "Active Learning test": "#2ca02c",     # Green
        "Random Sampling test": "#d62728",     # Red
        "Weighted Active Learning test": "#2ca02c", # green
        "Weighted Random Sampling test": "#d62728",       # red
        "Weighted Active Learning train": "#1f77b4", # blue
        "Weighted Random Sampling train": "#8c564b"       # brown
    }

    plt.plot(train_sizes_percent, avg_active_model_f1_plot, label="Active Learning CV", color=colors["Active Learning CV"])
    plt.plot(train_sizes_percent, avg_random_sampling_f1_plot, label="Random Sampling CV", color=colors["Random Sampling CV"])
    plt.plot(train_sizes_percent, active_test_model_f1, label="Active Learning test", color=colors["Active Learning test"])
    plt.plot(train_sizes_percent, random_test_sampling_f1, label="Random Sampling test", color=colors["Random Sampling test"])
    plt.plot(train_sizes_percent, weighted_active_f1, label="Weighted Active Learning test", color=colors["Weighted Active Learning test"], linestyle='--')
    plt.plot(train_sizes_percent, weighted_random_f1, label="Weighted Random Sampling test", color=colors["Weighted Random Sampling test"], linestyle='--')
    plt.plot(train_sizes_percent, weighted_f1_train_active, label="Weighted Active Learning train", color=colors["Weighted Active Learning train"], linestyle='--')
    plt.plot(train_sizes_percent, weighted_f1_train_random, label="Weighted Random Sampling train", color=colors["Weighted Random Sampling train"], linestyle='--')
    
    plt.xlabel("Train Size (%)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Train Size")
    plt.legend()
    plt.grid(True)
    #plt.xlim(left=10)
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


def plot_accuracies(train_sizes, active_learning_cv_accuracy, random_sampling_cv_accuracy, active_test_accuracy, random_test_accuracy):
    plt.figure()

    train_sizes_percent = np.array(train_sizes) * 100

    # Define a set of distinct colors
    colors = {
        "Active Learning CV": "#1f77b4",       # Blue
        "Random Sampling CV": "#8c564b",  # Brown
        "Active Learning test": "#2ca02c",     # Green
        "Random Sampling test": "#d62728"      # Red
    }

    plt.plot(train_sizes_percent, active_learning_cv_accuracy, label="Active Learning CV", color=colors["Active Learning CV"])
    plt.plot(train_sizes_percent, random_sampling_cv_accuracy, label="Random Sampling CV", color=colors["Random Sampling CV"])
    plt.plot(train_sizes_percent, active_test_accuracy, label="Active Learning test", color=colors["Active Learning test"])
    plt.plot(train_sizes_percent, random_test_accuracy, label="Random Sampling test", color=colors["Random Sampling test"])
    
    plt.xlabel("Train Size (%)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Train Size")
    plt.legend()
    plt.grid(True)


    plt.show()


def main():

    # # Ensure NLTK data is available
    # nltk.download('punkt')

    # # Load and preprocess the dataset
    # dataset = preprocess_and_vectorize(DATA_FILE)

    # Load and preprocess the dataset
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", header=None)
    dataset = df.values

    # Impute missing values with the mean of each column
    imputer = SimpleImputer(strategy="mean")
    dataset = imputer.fit_transform(dataset)

    # Standardize the dataset
    scaler = StandardScaler()
    dataset[:, :-1] = scaler.fit_transform(dataset[:, :-1])

    total_data_size = len(dataset)
    initial_train_size = int(INITIAL_TRAIN_SIZE_PERCENTAGE * total_data_size)

    train_sizes = []
    avg_active_model_f1_plot = []
    avg_random_sampling_f1_plot = []
    active_test_model_f1 = []
    random_test_sampling_f1 = []
    weighted_active_f1 = []
    weighted_random_f1 = []
    
    weighted_f1_train_active = []
    weighted_f1_train_random = []

    avg_active_model_accuracy_plot = []
    avg_random_sampling_accuracy_plot = []
    active_test_model_accuracy = []
    random_test_sampling_accuracy = []

    x_train, y_train, x_pool, y_pool = split_dataset_initial(dataset, initial_train_size / total_data_size)
    unlabel, label, x_test, y_test = split_pool_set(x_pool, y_pool, TEST_SIZE)

    print(f"x_train shape: {x_train.shape}")
    print(f"x_train shape: {y_train.shape}")
    print("#####################")
    print(f"x_train shape: {x_pool.shape}")
    print(f"x_train shape: {y_pool.shape}")
    print("#####################")
    print(f"x_train shape: {unlabel.shape}")
    print(f"x_train shape: {label.shape}")
    print("#####################")
    print(f"x_train shape: {x_test.shape}")
    print(f"x_train shape: {y_test.shape}")
    print("#####################")

    # # Compute GMM-based weights for the training set
    # train_weights = compute_gmm_weights_for_train(x_train, y_train)

    # Compute GMM-based weights for the test set
    test_weights = compute_gmm_weights(x_test, y_test)
    train_weights = compute_gmm_weights_for_train(unlabel, label)

    num_plots = int(1 / 0.1) + 1
    fig, axes = plt.subplots(num_plots, 2, figsize=(14, 7 * num_plots))

    plot_index = 0

    for train_size_percentage in TRAIN_SIZE_RANGE:
        try:

    
            train_size = int(train_size_percentage * total_data_size)

            x_train_active, y_train_active, unlabel_active, label_active = active_learning(
                x_train.copy(), y_train.copy(), unlabel.copy(), label.copy(), train_size, total_data_size)

            # Compute weights for the active learning training set
            #weights op totaal train set

            # print(f"x_train shape: {x_train.shape}")
            # print(f"x_train_active shape: {x_train_active.shape}")

            train_weights_active = train_weights[:len(y_train_active)]
            

            # train_weights_active = compute_gmm_weights_for_train(x_train_active, y_train_active)
            

            # Evaluate active learning model with weights
            avg_accuracy, active_f1, _, _, _ = evaluate_model_with_cross_validation(x_train_active, y_train_active)
            avg_active_model_f1_plot.append(active_f1)
            avg_active_model_accuracy_plot.append(avg_accuracy)

            classifier_active = train_model(x_train_active, y_train_active)
            test_accuracy, active_test_f1, _, _, _ = evaluate_model_on_test_set(classifier_active, x_test, y_test)
            active_test_model_f1.append(active_test_f1)
            active_test_model_accuracy.append(test_accuracy)

            # Compute weighted F1 for active learning
            _, weighted_f1_active, _, _, _ = evaluate_model_on_test_set_weighted(classifier_active, x_test, y_test, test_weights)
            weighted_active_f1.append(weighted_f1_active)

            # Compute weighted F1 for the training set
            train_accuracy, _, _, _, _ = evaluate_model_on_train_set_weighted(classifier_active, x_train_active, y_train_active, train_weights_active)
            weighted_f1_train_active.append(train_accuracy)

            # Random sampling for comparison
            rand_indices = np.random.choice(range(len(unlabel)), size=len(y_train_active), replace=False)
            x_rand_train = np.concatenate((x_train, unlabel[rand_indices]))
            y_rand_train = np.concatenate((y_train, label[rand_indices]))


            # Evaluate random sampling model with weights
            avg_accuracy, rand_f1, _, _, _ = evaluate_model_with_cross_validation(x_rand_train, y_rand_train)
            avg_random_sampling_f1_plot.append(rand_f1)
            avg_random_sampling_accuracy_plot.append(avg_accuracy)

            classifier_rand = train_model(x_rand_train, y_rand_train)
            test_accuracy, random_test_f1, _, _, _ = evaluate_model_on_test_set(classifier_rand, x_test, y_test)
            random_test_sampling_f1.append(random_test_f1)
            random_test_sampling_accuracy.append(test_accuracy)

            # Compute weighted F1 for random sampling
            _, weighted_f1_rand, _, _, _ = evaluate_model_on_test_set_weighted(classifier_rand, x_test, y_test, test_weights)
            weighted_random_f1.append(weighted_f1_rand)

            #IT SEEMS THE RANDOM SET IS 1% AHEAD
            train_weights_rand = train_weights[:len(y_rand_train)]

            print(f"  x_rand_train size: {len(x_rand_train)}")
            print(f"  y_rand_train size: {len(y_rand_train)}")
            print(f"  Unlabel size: {len(unlabel)}")
            print(f"  Rand indices size: {len(rand_indices)}")

            # Compute weighted F1 for the random sampling training set
            train_accuracy_rand, _, _, _, _ = evaluate_model_on_train_set_weighted(classifier_rand, x_rand_train, y_rand_train, train_weights_rand)
            weighted_f1_train_random.append(train_accuracy_rand)

            train_sizes.append(train_size_percentage)

            # Fit Gaussian Mixture Model to calculate densities and plot
            weights, gmm_train_active, gmm_test_active = fit_and_plot_gmm_density(x_train_active, y_train_active, x_test, y_test)

            if train_size_percentage * 100 % 10 == 0:
                plot_gmm_ellipsoids(gmm_train_active, x_train_active, axes[plot_index, 0], f'Active Learning ({train_size_percentage*100:.0f}%)')
                plot_gmm_ellipsoids(gmm_test_active, x_test, axes[plot_index, 1], 'Test Set')
                plot_index += 1

        except Exception as e:
            print(f"An error occurred during processing for train size {train_size_percentage*100:.2f}%: {e}")
            continue

    plot_gmm_ellipsoids(gmm_train_active, x_train_active, axes[plot_index, 0], f'Active Learning (Final)')
    plot_gmm_ellipsoids(gmm_test_active, x_test, axes[plot_index, 1], 'Test Set (Final)')

    plt.tight_layout()
    plt.show()

    # Plot F1 Scores
    plot_f1_scores(train_sizes, avg_active_model_f1_plot, avg_random_sampling_f1_plot, active_test_model_f1, random_test_sampling_f1, weighted_active_f1, weighted_random_f1, weighted_f1_train_active, weighted_f1_train_random)

    # Plot Accuracies
    plot_accuracies(train_sizes, avg_active_model_accuracy_plot, avg_random_sampling_accuracy_plot, active_test_model_accuracy, random_test_sampling_accuracy)

    # Plot ROC Curve
    plt.figure()
    plot_roc_curve(y_test, classifier_active.predict_proba(x_test)[:, 1], "Active Learning")
    plot_roc_curve(y_test, classifier_rand.predict_proba(x_test)[:, 1], "Random Sampling")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure()
    plot_precision_recall_curve(y_test, classifier_active.predict_proba(x_test)[:, 1], "Active Learning")
    plot_precision_recall_curve(y_test, classifier_rand.predict_proba(x_test)[:, 1], "Random Sampling")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print weighted F1 scores for training set
    print("Weighted F1 Scores for Training Set:")
    for idx, train_size_percentage in enumerate(TRAIN_SIZE_RANGE):
        print(f"Train Size: {train_size_percentage*100:.0f}%")
        print(f"  Active Learning Weighted F1: {weighted_f1_train_active[idx]:.4f}")
        print(f"  Random Sampling Weighted F1: {weighted_f1_train_random[idx]:.4f}")
        print()

if __name__ == "__main__":
    main()
