from sklearn.linear_model import LogisticRegression

def create_logreg(C=1.0, max_iter=2000, class_weight='balanced', random_state=42):
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state
    )