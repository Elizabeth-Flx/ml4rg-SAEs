import numpy as np
from sklearn import metrics
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

def normalize_classifier_matrix(X):
    """
    column-wise, should normalize by prediction
    i.e. rows correspond to observations 
    and columns correspond to predictions
    """
    if X.min()>=0 and X.max()<=1:
        return X
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # Avoid division by zero
    return (X - min_vals) / ranges

def normalize_binary(ground_truth):
    if np.array_equal(ground_truth, ground_truth.astype(bool)):
        return ground_truth
    return [0 if x == 0 else 1 for x in ground_truth]

# INPUT: Vector, binary baseline
# OUTPUT: AUC
def calculate_AUC_vector(prediction, ground_truth):
    """
    prediction: 1D vector, range 0-1
    ground_truth: 1D vector, 0 or 1, or binary
    
    output: AUC score
    """
    return metrics.roc_auc_score(ground_truth, prediction)

# INPUT: Vector, binary baseline
# OUTPUT: average precision
def calculate_precision_vector(prediction, ground_truth):
    """
    prediction: 1D vector, range 0-1
    ground_truth: 1D vector, 0 or 1, or binary
    
    output: average precision
    """
    return metrics.average_precision_score(ground_truth, prediction, pos_label=1)

def calculate_AUC_matrix(prediction_matrix, ground_truth_matrix):
    """
    AUC for each prediction  * each  ground_truth

    prediction_matrix of shape: (#obs, #predicted features); gets each predicted feature gets min-max  to range 0-1.
    ground_truth_matrix of shape: (#obs, #ground truth features)

    output of shape(#predicted features, #ground truth features)
    """

    prediction_matrix = normalize_classifier_matrix(prediction_matrix).T
    ground_truth_matrix = normalize_binary(ground_truth_matrix).T

    n_predictions, n_pred_observations = np.shape(prediction_matrix)
    n_truths, n_truth_observations = np.shape(ground_truth_matrix)

    assert n_pred_observations == n_truth_observations

    AUCs = np.empty((n_predictions, n_truths))

    p = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    )
    with p:
        t = p.add_task("AUC", total=n_predictions)
        for i, prediction in enumerate(prediction_matrix):
            for j, ground_truth in enumerate(ground_truth_matrix):
                AUC = calculate_AUC_vector(prediction=prediction, ground_truth=ground_truth)
                AUCs[i,j] = AUC
            p.update(t, advance=1)
    return AUCs

def calculate_precision_matrix(prediction_matrix, ground_truth_matrix):
    """
    average precision for each prediction * each  ground_truth

    prediction_matrix of shape: (#obs, #predicted features); gets each predicted feature gets min-max  to range 0-1.
    ground_truth_matrix of shape: (#obs, #ground truth features)

    output of shape(#predicted features, #ground truth features)
    """
    prediction_matrix = normalize_classifier_matrix(prediction_matrix).T
    ground_truth_matrix = normalize_binary(ground_truth_matrix).T

    n_predictions, n_pred_observations = np.shape(prediction_matrix)
    n_truths, n_truth_observations = np.shape(ground_truth_matrix)

    assert n_pred_observations == n_truth_observations

    precisions = np.empty((n_predictions, n_truths))
    p = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    )
    with p:
        t = p.add_task("precision", total=n_predictions)
        for i, prediction in enumerate(prediction_matrix):
            for j, ground_truth in enumerate(ground_truth_matrix):
                precision = calculate_precision_vector(prediction=prediction, ground_truth=ground_truth)
                precisions[i,j] = precision
            p.update(t, advance=1)
    return precisions