from sklearn.svm import SVC

def create_svm(kernel='rbf', random_state=42):
    return SVC(kernel=kernel, random_state=random_state)