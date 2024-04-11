import numpy as np
import pandas as pd
import argparse
from metrics.metrics import calculate_roc_auc, find_best_threshold

def load_probabilities(directory, dataset_type):
    """
    Load probabilities and labels for a given dataset type ('val' or 'test').

    Args:
        directory (str): Path to the directory containing the probabilities and labels.
        dataset_type (str): The type of the dataset ('val' or 'test').

    Returns:
        tuple: A tuple containing the loaded probabilities and labels numpy arrays.
    """
    probabilities = np.load(f"{directory}/probabilities_{dataset_type}.npy")
    labels = np.load(f"{directory}/labels_{dataset_type}.npy")
    return probabilities, labels

def predict_and_evaluate(probabilities_dir, data_frame_path, save_path, prediction_name =''):
    """
    Pipeline for predicting and evaluating the model on the test set and saving the results.

    Args:
        probabilities_dir (str): Directory containing the probabilities and labels for the datasets.
        data_frame_path (str): Path to the CSV file with test data.
        save_path (str): Path where the final DataFrame should be saved.

    Returns:
        None: The function will save two CSV files with predictions and probabilities for each label, and evaluation metrics.
    """
    # Define prediction labels
    if prediction_name == '':
        prediction_name = probabilities_dir.split('/')[-2]
        print(f"Prediction name not provided. Using {prediction_name} as prediction name")
    prediction_labels = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
        'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]

    # Load probabilities and labels for validation and test datasets
    val_prob, val_labels = load_probabilities(probabilities_dir, 'val')
    test_prob, test_labels = load_probabilities(probabilities_dir, 'test')

    # Calculate ROC AUC
    _, _, class_auc = calculate_roc_auc(test_prob, test_labels)


    # Find the best threshold
    best_threshold = find_best_threshold(val_prob, val_labels)

    # Calculate predictions on the test set
    test_pred = np.zeros_like(test_prob)
    for i, label in enumerate(prediction_labels):
        test_pred[:, i] = (test_prob[:, i] > best_threshold[i]).astype(int)

    # Create a DataFrame with predictions and probabilities for each label
    df = pd.read_csv(data_frame_path)
    for i, label in enumerate(prediction_labels):
        df[f"bi_{label}"] = test_pred[:, i]
        df[f"prob_{label}"] = test_prob[:, i]
    
    # Save the DataFrame with predictions and probabilities
    df.to_csv(f"{save_path}/{prediction_name}_predictions_and_probabilities.csv", index=False)

    # Save evaluation metrics
    df_info = pd.DataFrame({
        'label': prediction_labels,
        'class_auc': list(class_auc.values()),
        'best_threshold': [best_threshold[label] for label in range(len(prediction_labels))]
    })
    df_info.to_csv(f"{save_path}/{prediction_name}_evaluation_metrics.csv", index=False)

    print(f"Files saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and Evaluate the Model")
    parser.add_argument("probabilities_dir", type=str, help="Directory containing the probabilities and labels")
    parser.add_argument("data_frame_path", type=str, help="Path to the CSV file with test data")
    parser.add_argument("save_path", type=str, help="Path where to save the final DataFrame")

    args = parser.parse_args()

    predict_and_evaluate(args.probabilities_dir, args.data_frame_path, args.save_path)