import numpy as np
from sklearn import metrics

def normalize_classifier_score(prediction):
    if np.min(prediction) >= 0 and np.max(prediction) <= 1:
        return prediction 
    return (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
def normalize_binary(ground_truth):
    return [0 if x == 0 else 1 for x in ground_truth]

# INPUT: Vector, binary baseline
# OUTPUT: AUC
def calculate_AUC_vector(prediction, ground_truth):
    """
    prediction: 1D vector, will get normalized to range 0-1 if it isn't already (min-max-normalization, except if range already fits)
    ground_truth: 1D vector, will get normalized to range 0-1 if it isn't already (with everything above 0 -> 1)
    
    output: AUC score
    """
    ground_truth = normalize_binary(ground_truth)
    prediction = normalize_classifier_score(prediction)
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth, prediction, pos_label=1) # positive: 1
    return metrics.auc(fpr, tpr)

# INPUT: Vector, binary baseline
# OUTPUT: average precision
def calculate_precision_vector(prediction, ground_truth):
    """
    prediction: 1D vector, will get normalized to range 0-1 if it isn't already (min-max-normalization, except if range already fits)
    ground_truth: 1D vector, will get normalized to range 0-1 if it isn't already (with everything above 0 -> 1)
    
    output: average precision
    """
    ground_truth = normalize_binary(ground_truth)
    prediction = normalize_classifier_score(prediction)
    return metrics.average_precision_score(ground_truth, prediction, pos_label=1)

def calculate_AUC_matrix(prediction_matrix, ground_truth_matrix):
    """
    AUC for each row in prediction  *  each row in ground_truth
    returns: matrix of AUCs; each row corresponds to one predicted feature, and columns correspond to the ground truth
    """
    
    n_predictions, length_predictions = np.shape(prediction_matrix)
    n_truths, length_truths = np.shape(ground_truth_matrix)

    assert length_predictions == length_truths
    
    AUCs = np.empty((n_predictions, n_truths))
    for i, prediction_row in enumerate(prediction_matrix):
        for j, ground_truth_row in enumerate(ground_truth_matrix):
            AUC = calculate_AUC_vector(prediction=prediction_row, ground_truth=ground_truth_row)
            AUCs[i,j] = AUC
    return AUCs

def calculate_precision_matrix(prediction_matrix, ground_truth_matrix):
    """
    precision for each row in prediction  *  each row in ground_truth
    returns: matrix of average precisions; each row corresponds to one predicted feature, and columns correspond to the ground truth
    """
    
    n_predictions, length_predictions = np.shape(prediction_matrix)
    n_truths, length_truths = np.shape(ground_truth_matrix)

    assert length_predictions == length_truths
    
    precisions = np.empty((n_predictions, n_truths))
    for i, prediction_row in enumerate(prediction_matrix):
        for j, ground_truth_row in enumerate(ground_truth_matrix):
            precision = calculate_precision_vector(prediction=prediction_row, ground_truth=ground_truth_row)
            precisions[i,j] = precision
    return precisions

preds = np.array([[0.3, 0.4, 0.5],[1, 0.9, 0.4], [0, 1, 0]])
truths = np.array([[0,1,1], [1,1,0], [1, 0, 0]])

print(calculate_AUC_matrix(preds, truths))
print(calculate_precision_matrix(preds, truths))