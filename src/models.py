from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300)
    }
