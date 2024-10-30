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
from scipy.interpolate import make_interp_spline
import random
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import seaborn as sns
import math
from matplotlib.gridspec import GridSpec
from scipy.stats import entropy
from sklearn.metrics import precision_score, log_loss, f1_score, accuracy_score
from sklearn.metrics import precision_score, log_loss, f1_score as sklearn_f1_score
from sklearn.metrics import confusion_matrix
import os


np.seterr(over='ignore')  # Suppress overflow warnings

# Constants
# DATA_FILE = 'SMSSpamCollection'
DATA_FILE = 'Sentiment_reformat'
INITIAL_TRAIN_SIZE_PERCENTAGE = 0.01
TRAIN_SIZE_RANGE = np.arange(0.02, 0.74, 0.01)
ACTIVE_LEARNING_ROUNDS = 1
N_SPLITS = 5
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
def split_dataset_initial(dataset, train_size, num):
    x = dataset[:, :-1]  # Features
    y = dataset[:, -1].astype(int)  # Labels as integers
    x_train, x_pool, y_train, y_pool = train_test_split(x, y, train_size=train_size, random_state=num, stratify=y)
    return x_train, y_train, x_pool, y_pool

def split_pool_set(x_pool, y_pool, test_size, num):
    unlabel, x_test, label, y_test = train_test_split(x_pool, y_pool, test_size=test_size, random_state=num, stratify=y_pool)
    return unlabel, label, x_test, y_test

# Function to train the model
def train_model(x_train, y_train):
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x_train, y_train)
    return classifier

def active_learning(x_train, y_train, unlabel, label, target_train_size, total_data_size):
    # Start by tracking indices in the initial labeled set
    y_train_indices = np.arange(len(y_train))
    # print("Initial y_train_indices:", y_train_indices)

    while len(y_train) < target_train_size:
        classifier = train_model(x_train, y_train)

        if len(unlabel) == 0:
            print("No more samples to select from. Stopping active learning.")
            break
        
        # Model predictions and uncertainty calculations
        y_probab = classifier.predict_proba(unlabel)[:, 1]
        uncertainties = np.abs(y_probab - 0.5)
        uncertain_indices = np.argsort(uncertainties)[:min(len(uncertainties), target_train_size - len(y_train))]

        if len(uncertain_indices) == 0:
            break

        # Adding uncertain samples to x_train and y_train
        x_train = np.append(unlabel[uncertain_indices, :], x_train, axis=0)
        y_train = np.append(label[uncertain_indices], y_train)
        unlabel = np.delete(unlabel, uncertain_indices, axis=0)
        label = np.delete(label, uncertain_indices)

        # Use uncertain_indices directly as the new indices from the unlabel set
        # print("\nIndices of newly selected samples from unlabel:", uncertain_indices)
        
        # Append the original indices of new samples from unlabel to y_train_indices
        y_train_indices = np.append(uncertain_indices, y_train_indices)
        # print("Updated y_train_indices:", y_train_indices)

        current_train_percentage = (len(y_train) / total_data_size) * 100
        # print(f"Total train size: {len(y_train)}, data used: {current_train_percentage:.2f}%")

    return x_train, y_train, unlabel, label, y_train_indices



# Function to evaluate the model on the test set
def evaluate_model_on_test_set(classifier, x_test, y_test):
    accuracy = classifier.score(x_test, y_test)
    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)
    return (
        accuracy,
        f1_score(y_test, y_pred, zero_division=1),
        precision_score(y_test, y_pred, zero_division=1),
        log_loss(y_test, y_prob),
        y_prob,
    )

# Function to evaluate the model on the test set with weighted F1 score
def evaluate_model_on_test_set_weighted(classifier, x_test, y_test):
    y_pred = classifier.predict(x_test)
    f1_weighted = f1_score(y_test, y_pred, zero_division=1)
    accuracy = classifier.score(x_test, y_test)
    y_prob = classifier.predict_proba(x_test)
    return (
        accuracy,
        f1_weighted,
        precision_score(y_test, y_pred, zero_division=1),
        log_loss(y_test, y_prob),
        y_prob,
    )

# Function to compute GMM-based log-probabilities on the training set
def compute_kde_log_weights_for_train(x_train, y_train):
    # gmm = GaussianMixture(n_components=2, random_state=42)
    # gmm.fit(x_train)
    # print_debug_info('X_TRAIN;', x_train)

    gmm = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x_train[:,0:2])
    
    # Compute the log likelihood for each sample
    log_likelihoods = gmm.score_samples(x_train[:,0:2])


    weights=log_likelihoods

    
    return weights    # To assign weights based on class labels


def evaluate_model_on_train_set_weighted(classifier, x_train, y_train, weights, n_splits=5):
    classifier = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies, precision_scores, losses = [], [], []
    all_y_val = []
    all_y_pred = []
    all_val_weights = []

    for train_index, val_index in skf.split(x_train, y_train):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        val_weights = weights[val_index]  # Use weights for the validation fold

        # Train the model
        classifier.fit(x_train_fold, y_train_fold)

        # Predict and calculate probabilities
        y_pred = classifier.predict(x_val_fold)
        y_prob = classifier.predict_proba(x_val_fold)[:, 1]  # Probability for class 1

        # Ensure correct data types
        y_val_fold = np.array(y_val_fold, dtype=int)
        y_pred = np.array(y_pred, dtype=int)

        # Collect all predictions and ground truth
        all_y_val.extend(y_val_fold)
        all_y_pred.extend(y_pred)
        all_val_weights.extend(val_weights)

        # Calculate metrics per fold
        accuracies.append(accuracy_score(y_val_fold, y_pred, sample_weight=val_weights))  # Weighted accuracy
        precision_scores.append(precision_score(y_val_fold, y_pred, zero_division=1, sample_weight=val_weights))  # Weighted precision
        losses.append(log_loss(y_val_fold, y_prob, sample_weight=val_weights))  # Weighted log loss

    # Convert collected lists to numpy arrays
    all_y_val = np.array(all_y_val)
    all_y_pred = np.array(all_y_pred)
    all_val_weights = np.array(all_val_weights)

    # Calculate final F1 score using the entire validation set
    f1_final = f1_score(all_y_val, all_y_pred, sample_weight=all_val_weights)

    # Return averaged results across folds and the final F1 score
    return np.mean(accuracies), f1_final, np.mean(precision_scores), np.mean(losses), classifier



# Function to plot ROC curve
def plot_roc_curve(y_true, y_probs, label):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(y_true, y_probs, label):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(recall, precision, label=f'{label} (PR Curve)')

def smooth_line(x, y, num_points=300):
    x_smooth = np.linspace(x.min(), x.max(), num_points)  # More points for smooth line
    spline = make_interp_spline(x, y, k=3)  # Cubic spline interpolation
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

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

    # Before plotting, check the values of x and y
    print("train_sizes_percent (x):", train_sizes_percent)
    
    if isinstance(train_sizes_percent, list):
        print("Length of train_sizes_percent:", len(train_sizes_percent))
    if isinstance(weighted_f1_train_active, list):
        print("Length of weighted_f1_train_active:", len(weighted_f1_train_active))

    plt.plot(train_sizes_percent, avg_active_model_f1_plot, label="Active Learning CV", color=colors["Active Learning CV"])
    plt.plot(train_sizes_percent, avg_random_sampling_f1_plot, label="Random Sampling CV", color=colors["Random Sampling CV"])
    plt.plot(train_sizes_percent, active_test_model_f1, label="Active Learning test", color=colors["Active Learning test"])
    plt.plot(train_sizes_percent, random_test_sampling_f1, label="Random Sampling test", color=colors["Random Sampling test"])
    # plt.plot(train_sizes_percent, weighted_active_f1, label="Weighted Active Learning test", color=colors["Weighted Active Learning test"], linestyle='--')
    # plt.plot(train_sizes_percent, weighted_random_f1, label="Weighted Random Sampling test", color=colors["Weighted Random Sampling test"], linestyle='--')
    plt.plot(train_sizes_percent, weighted_f1_train_active, label="Weighted Active Learning train", color=colors["Weighted Active Learning train"], linestyle='--')
    # plt.plot(train_sizes_percent, weighted_f1_train_random, label="Weighted Random Sampling train", color=colors["Weighted Random Sampling train"], linestyle='--')

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
def plot_gmm_ellipsoids(gmm, X, ax=None, label='GMM Components'):
    if ax is None:
        fig, ax = plt.subplots()  # Create a new figure and axes if none are provided

    colors = ['red', 'blue']  # Adjust the colors based on the number of components
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        
        ell = Ellipse(
            xy=mean,
            width=v[0],
            height=v[1],
            angle=np.degrees(np.arctan2(u[1], u[0])),
            color=colors[i % len(colors)],
            alpha=0.5,
            label=f'{label} Component {i+1}'
        )
        ax.add_patch(ell)

    ax.scatter(X[:, 0], X[:, 1], s=10, color='black', label='Data')
    ax.set_title(f'GMM Ellipsoids - {label}')
    ax.legend()
    plt.show()



def print_debug_info(name, data):
    if isinstance(data, (list, np.ndarray)):
        print(f"{name} - Type: {type(data)}")
        print(f"{name} - Size: {len(data)}")
        if isinstance(data, np.ndarray):
            print(f"{name} - Shape: {data.shape}")
        else:
            print(f"{name} - Length: {len(data)}")
        print(f"{name} - Sample (first 5 elements): {data[:5]}")
    else:
        print(f"{name} - Type: {type(data)}")
        print(f"{name} - Value: {data}")


# Function to save the joint KDE plot and show data points colored by class
def plot_joint_kde(x_train, y_train, save_dir="plots", file_name="joint_kde_plot"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a figure with a specific layout for marginal distributions
    fig = plt.figure(figsize=(12, 12))  # Keep figure size square
    gs = GridSpec(5, 4, wspace=0.5, hspace=1.2)  # Increase hspace to add vertical space between plots

    # Create the main KDE plot with data points
    ax_main = fig.add_subplot(gs[1:3, 0:3])  # Main plot occupies the center
    sns.kdeplot(x=x_train[:, 0], y=x_train[:, 1], fill=True, ax=ax_main, cmap="Blues", alpha=0.5)

    # Overlay scatter plot with different colors for each class
    classes = np.unique(y_train)
    colors = ["r", "g"]  # Red for spam, Green for not spam

    for i, cls in enumerate(classes):
        # Plot points for each class
        mask = y_train == cls
        class_label = "spam" if cls == 0 else "not spam"
        ax_main.scatter(x_train[mask, 0], x_train[mask, 1], label=class_label, color=colors[i], edgecolor="black")

    # Add axis labels and title for the main plot
    ax_main.set_xlabel("Feature 1")
    ax_main.set_ylabel("Feature 2")
    ax_main.set_title("KDE with Data Points")
    ax_main.legend(title="Classes")

    # Create marginal distributions
    ax_x = fig.add_subplot(gs[0, 0:3])  # Top marginal
    ax_y = fig.add_subplot(gs[1:3, 3])  # Right marginal

    # Plotting marginal distribution for Feature 1 (top)
    sns.kdeplot(x=x_train[:, 0], ax=ax_x, color='blue', fill=True, alpha=0.5)
    ax_x.set_ylabel("Density")
    ax_x.set_title("Marginal Distribution of Feature 1")
    ax_x.set_xticks([])  # Hide x-ticks on marginal plot

    # Plotting marginal distribution for Feature 2 (right, flipped 90 degrees)
    sns.kdeplot(y=x_train[:, 1], ax=ax_y, color='blue', fill=True, alpha=0.5)  # Flip axis by using y=
    ax_y.set_aspect('auto')  # Allow plot to stretch horizontally
    ax_y.set_xlabel("Density")
    ax_y.set_title("Marginal Distribution of Feature 2 (Flipped)")
    ax_y.set_yticks([])  # Hide y-ticks on marginal plot

    # Create the second KDE plot without data points, making it square
    ax_kde_without_data = fig.add_subplot(gs[3:, 0:3])  # Bottom plot spans 2 rows, 3 columns
    sns.kdeplot(x=x_train[:, 0], y=x_train[:, 1], fill=True, ax=ax_kde_without_data, cmap="Blues", alpha=0.5)

    # Add axis labels and title for the second plot
    ax_kde_without_data.set_xlabel("Feature 1")
    ax_kde_without_data.set_ylabel("Feature 2")
    ax_kde_without_data.set_title("Larger KDE without Data Points")

    # Adjust layout with padding to move the bottom plot down
    plt.subplots_adjust(bottom=0.1, top=0.95)  # Increase bottom space

    # Save the plot instead of showing it
    save_path = os.path.join(save_dir, f"{file_name}.png")
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure to file
    plt.close()  # Close the figure to free memory

    print(f"Saved KDE plot to {save_path}")


# Function to save the simple joint KDE plot
def plot_joint_kde_simple(x_train, y_train, save_dir="plots", file_name="joint_kde_simple_plot"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Combine the training data (x_train and y_train) into a single array for plotting
    data = np.column_stack((x_train, y_train))

    # Assuming x_train has 2 features, you can create a jointplot
    jointplot = sns.jointplot(x=data[:, 0], y=data[:, 1], kind="kde", fill=True)

    # Save the plot instead of showing it
    save_path = os.path.join(save_dir, f"{file_name}.png")
    jointplot.savefig(save_path, bbox_inches='tight')  # Save the jointplot to file
    plt.close()  # Close the figure to free memory

    print(f"Saved simple KDE plot to {save_path}")


def run_experiment(dataset, total_data_size, experiments):
    """
    Runs multiple cycles of experiments and aggregates the results.
    """

    initial_train_size = int(INITIAL_TRAIN_SIZE_PERCENTAGE * total_data_size)
    
    cumulative_results = {
        'avg_active_f1': [0] * len(TRAIN_SIZE_RANGE),
        'random_sampling_f1': [0] * len(TRAIN_SIZE_RANGE),
        'active_test_f1': [0] * len(TRAIN_SIZE_RANGE),
        'random_test_f1': [0] * len(TRAIN_SIZE_RANGE),
        'weighted_active_f1': [0] * len(TRAIN_SIZE_RANGE),
        'weighted_random_f1': [0] * len(TRAIN_SIZE_RANGE),
        'weighted_train_active_f1': [0] * len(TRAIN_SIZE_RANGE),
        'weighted_train_random_f1': [0] * len(TRAIN_SIZE_RANGE),
    }
    
    # Run multiple experiments
    for _ in range(experiments):
        train_sizes, results, x_test, y_test, classifier_active, classifier_rand = run_training_cycle(dataset, initial_train_size, total_data_size)

        # Aggregate the results
        for key in cumulative_results:
            cumulative_results[key] = [x + y for x, y in zip(cumulative_results[key], results[key])]
    
    # Average the results over the number of experiments
    for key in cumulative_results:
        cumulative_results[key] = [x / experiments for x in cumulative_results[key]]
    
    return train_sizes, cumulative_results, x_test, y_test, classifier_active, classifier_rand


def run_training_cycle(dataset, initial_train_size, total_data_size):
    """
    Runs a single experiment cycle for active learning and random sampling.
    Uses the weights during training and evaluation.
    """
    # Shuffle dataset
    random.shuffle(dataset)

    train_sizes = []
    
    # Initialize lists to store metrics for this experiment
    avg_active_f1_list = []
    avg_random_f1_list = []
    active_test_f1_list = []
    random_test_f1_list = []
    weighted_active_f1_list = []
    weighted_random_f1_list = []
    weighted_f1_train_active_list = []
    weighted_f1_train_random_list = []

    # Split dataset into training, pool, and test sets
    num = random.randint(1, 500)
    x_train, y_train, x_pool, y_pool = split_dataset_initial(dataset, initial_train_size / total_data_size, num)
    unlabel, label, x_test, y_test = split_pool_set(x_pool, y_pool, TEST_SIZE, num)

    # Assuming x_train and y_train are already defined
    plot_joint_kde(x_train, y_train, save_dir="plots", file_name="initial_joint_kde")
    plot_joint_kde_simple(x_train, y_train, save_dir="plots", file_name="initial_joint_kde_simple")
    plot_joint_kde(unlabel, label, save_dir="plots", file_name="full_joint_kde")
    plot_joint_kde_simple(unlabel, label, save_dir="plots", file_name="full_joint_kde_simple")



    # Compute GMM-based weights for the unlabelled data
    train_weights = compute_kde_log_weights_for_train(unlabel, label)
    # Get the shape of the existing array
    shape = train_weights.shape

    # Create a new array of ones with the same shape
    dummy_weights_whole_set = np.ones(shape)
    

    runs = 1  # Number of runs for each train size percentage
    for train_size_percentage in TRAIN_SIZE_RANGE:
        weighted_f1_train_active_list_run = []
        weighted_f1_train_random_list_run = []
        avg_active_model_f1_plot = []
        avg_random_sampling_f1_plot = []
        active_test_model_f1 = []
        random_test_sampling_f1 = []
        weighted_active_f1 = []
        weighted_random_f1 = []

        for _ in range(runs):
            try:
                train_size = int(train_size_percentage * total_data_size)
        
                
                # Active learning process
                x_train_active, y_train_active, _, _, y_train_indices = active_learning(
                    x_train.copy(), y_train.copy(), unlabel.copy(), label.copy(), train_size, total_data_size)
                
                if any(lower <= train_size_percentage * 100 <= upper for lower, upper in [(9.95, 10.05), (19.95, 20.05), 
                                                                                 (29.95, 30.05), (39.95, 40.05),
                                                                                 (49.95, 50.05), (59.95, 60.05),
                                                                                 (69.95, 70.05), (79.95, 80.05),
                                                                                 (89.95, 90.05), (99.95, 100.05)]):
                    plot_joint_kde(x_train_active, y_train_active, save_dir="plots", file_name=f"joint_kde_{train_size_percentage:.2f}")
                    plot_joint_kde_simple(x_train_active, y_train_active, save_dir="plots", file_name=f"joint_kde_simple_{train_size_percentage:.2f}")
  

                # Compute log weights for active learning
                train_weights_active_loop = compute_kde_log_weights_for_train(x_train_active, y_train_active)

                train_weights_active = np.exp(train_weights[y_train_indices] - train_weights_active_loop)

                # train_weights_active = np.exp(train_weights[:len(y_train_active)] - train_weights_active_loop)

                # Get the shape of the existing array
                shape = train_weights_active.shape

                # Save full array to file without truncation
                with open('./weights_loop.txt', 'a') as file:
                    file.write(np.array2string(train_weights_active, threshold=np.inf))
                    file.write('\n\n')

                # Create a new array of ones with the same shape
                dummy_weights = np.ones(shape)

                # Evaluate active learning model

                classifier_active = train_model(x_train_active, y_train_active)

                _, active_f1, _, _, _ = evaluate_model_on_train_set_weighted(
                    classifier_active, x_train_active, y_train_active, dummy_weights
                )
                avg_active_model_f1_plot.append(active_f1)

                print(active_f1)

                test_accuracy, active_test_f1, _, _, _ = evaluate_model_on_test_set(classifier_active, x_test, y_test)
                active_test_model_f1.append(active_test_f1)

                # Compute weighted F1 for active learning
                _, weighted_f1_active, _, _, _ = evaluate_model_on_test_set_weighted(classifier_active, x_test, y_test)
                weighted_active_f1.append(weighted_f1_active)

                _, weighted_f1_train_active, _, _, _ = evaluate_model_on_train_set_weighted(
                    classifier_active, x_train_active, y_train_active, train_weights_active
                )
                weighted_f1_train_active_list_run.append(weighted_f1_train_active)

                # Random sampling process for comparison
                rand_indices = np.random.choice(range(len(unlabel)), size=len(y_train_active), replace=False)
                x_rand_train = np.concatenate((x_train, unlabel[rand_indices]))
                y_rand_train = np.concatenate((y_train, label[rand_indices]))


                classifier_rand = train_model(x_rand_train, y_rand_train)
                avg_accuracy, rand_f1, _, _, _ = evaluate_model_on_train_set_weighted(classifier_rand, x_rand_train, y_rand_train, dummy_weights_whole_set[:len(y_rand_train)])
                avg_random_sampling_f1_plot.append(rand_f1)

                _, random_test_f1, _, _, _ = evaluate_model_on_test_set(classifier_rand, x_test, y_test)
                random_test_sampling_f1.append(random_test_f1)

                # Compute weighted F1 for random sampling
                _, weighted_f1_rand, _, _, _ = evaluate_model_on_test_set_weighted(classifier_rand, x_test, y_test)
                weighted_random_f1.append(weighted_f1_rand)

                _, weighted_f1_train_random, _, _, _ = evaluate_model_on_train_set_weighted(
                    classifier_rand, x_rand_train, y_rand_train, train_weights[:len(y_rand_train)]
                )
                weighted_f1_train_random_list_run.append(weighted_f1_train_random)

            except Exception as e:
                print(f"Error during processing for train size {train_size_percentage*100:.2f}%: {e}")
                continue

        # Store averaged results for this train size
        train_sizes.append(train_size_percentage)
        avg_active_f1_list.append(np.mean(avg_active_model_f1_plot))
        avg_random_f1_list.append(np.mean(avg_random_sampling_f1_plot))
        active_test_f1_list.append(np.mean(active_test_model_f1))
        random_test_f1_list.append(np.mean(random_test_sampling_f1))
        weighted_active_f1_list.append(np.mean(weighted_active_f1))
        weighted_random_f1_list.append(np.mean(weighted_random_f1))
        weighted_f1_train_active_list.append(np.mean(weighted_f1_train_active_list_run))
        weighted_f1_train_random_list.append(np.mean(weighted_f1_train_random_list_run))

    # Return results for this cycle
    results = {
        'avg_active_f1': avg_active_f1_list,
        'random_sampling_f1': avg_random_f1_list,
        'active_test_f1': active_test_f1_list,
        'random_test_f1': random_test_f1_list,
        'weighted_active_f1': weighted_active_f1_list,
        'weighted_random_f1': weighted_random_f1_list,
        'weighted_train_active_f1': weighted_f1_train_active_list,
        'weighted_train_random_f1': weighted_f1_train_random_list,
    }
    
    return train_sizes, results, x_test, y_test, classifier_active, classifier_rand

def log_results(results, log_file_path):
    """
    Logs the results to a specified file.
    """
    with open(log_file_path, 'w') as f:
        for key, values in results.items():
            f.write(f"{key}: {values}\n")

def log_results_append(results, log_file_path):
    """
    Logs the results to a specified file.
    """
    with open(log_file_path, 'a') as f:
        for key, values in results.items():
            f.write(f"{key}: {values}\n")
        f.write('\n\n')

def main(plot_results=True):
    # Overwrite the file at the start of the script to reset it
    with open('./weights_loop.txt', 'w') as file:
        file.write('')  # Writing an empty string will clear the file

    """
    Main function to process the dataset, perform PCA, and run experiments on active learning and random sampling.
    """
    n_components = 50

    # Ensure NLTK data is available
    nltk.download('punkt')

    # Load and preprocess the dataset
    dataset = preprocess_and_vectorize(DATA_FILE)

    # Impute missing values with the mean of each column
    imputer = SimpleImputer(strategy="mean")
    dataset = imputer.fit_transform(dataset)

    # Standardize the dataset (only for the feature columns, not the label)
    scaler = StandardScaler()
    dataset[:, :-1] = scaler.fit_transform(dataset[:, :-1])

    # Apply PCA to the dataset (excluding the label column)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(dataset[:, :-1])

    # Combine the selected principal components with the original labels
    new_dataset = np.hstack((pca_result, dataset[:, -1].reshape(-1, 1)))

    # Convert the new dataset to a pandas DataFrame for easier manipulation
    columns = [f'PC{i+1}' for i in range(n_components)] + ['label']
    pca_df = pd.DataFrame(new_dataset, columns=columns)

    print(f"New Dataset with the top {n_components} Principal Components:\n", pca_df)

    # Shuffle the dataset
    dataset = new_dataset
    random.shuffle(dataset)

    # Setup train/test size parameters
    total_data_size = len(dataset)

    experiments = 1
    cumulative_results = {key: [] for key in ['avg_active_f1', 'random_sampling_f1', 'active_test_f1', 'random_test_f1', 'weighted_active_f1', 'weighted_random_f1', 'weighted_train_active_f1', 'weighted_train_random_f1']}
    successful_experiments = 0  # Counter for successful experiments
    for _ in range(experiments):
        try:
            # Run experiment
            train_sizes, results, x_test, y_test, classifier_active, classifier_rand = run_experiment(dataset, total_data_size, experiments)

            # Check if any results contain NaN values
            if any(np.isnan(val) for result in results.values() for val in result):
                print("NaN value found in results, skipping this iteration.")
                continue  # Skip the current experiment if NaN is found

            # Store the results of the current experiment
            for key in cumulative_results:
                cumulative_results[key].append(results[key])
            
            # Increment successful experiments counter
            successful_experiments += 1

        except ValueError as e:
            print(f"ValueError encountered during the experiment: {e}")
            continue  # Continue to the next experiment

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue  # Continue to the next experiment

    # Final averages after all successful experiments
    if successful_experiments > 0:
        final_results = {key: np.mean(cumulative_results[key], axis=0).tolist() for key in cumulative_results}
    else:
        print("No successful experiments were conducted.")
        final_results = {}  # Or handle as needed

    # Log final results to file
    log_results(final_results, "./log.txt")

    print(f"Results saved to ./log.txt")

    print('Number of successful experiments:')
    print(successful_experiments)

    if plot_results:
        # Plot F1 Scores
        plot_f1_scores(
            train_sizes, 
            final_results['avg_active_f1'], 
            final_results['random_sampling_f1'], 
            final_results['active_test_f1'], 
            final_results['random_test_f1'], 
            final_results['weighted_active_f1'], 
            final_results['weighted_random_f1'], 
            final_results['weighted_train_active_f1'], 
            final_results['weighted_train_random_f1']
        )

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
        # plt.show()

    log_results_append(final_results, 'final_results.txt')

    # Print weighted F1 scores for training set
    print("Weighted F1 Scores for Training Set:")
    for idx, train_size_percentage in enumerate(TRAIN_SIZE_RANGE):
        print(f"Train Size: {train_size_percentage*100:.0f}%")
        print(f"  Active Learning Weighted F1: {final_results['weighted_train_active_f1'][idx]:.4f}")
        print(f"  Random Sampling Weighted F1: {final_results['weighted_train_random_f1'][idx]:.4f}")
        print()


if __name__ == "__main__":
    main(plot_results=True)  # You can pass False if you don't want to plot
