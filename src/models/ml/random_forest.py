from sklearn.ensemble import RandomForestClassifier

def create_random_forest(n_estimators=100, random_state=42, n_jobs=-1):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )