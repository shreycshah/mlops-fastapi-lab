from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the Breast Cancer dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the Breast Cancer dataset.
        y (numpy.ndarray): The target values of the Breast Cancer dataset.
    """
    data = load_breast_cancer()

    X = data.data
    # using just three main features for lab purposes
    feature_names = data.feature_names
    selected_features = ["mean radius", "mean texture", "mean smoothness"]
    feature_indices = [list(feature_names).index(f) for f in selected_features]
    X_selected = X[:, feature_indices]

    y = data.target

    return X_selected, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test