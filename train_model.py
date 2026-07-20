from sklearn.ensemble import RandomForestClassifier
import pandas as pd

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
print("Model created:", model)