from xgboost import XGBClassifier

def create_xgboost(n_estimators=100, random_state=42):
    return XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        eval_metric='mlogloss'
    )