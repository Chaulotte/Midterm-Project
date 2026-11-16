import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DATA_PATH = "data/penguins_size.csv"
OUTPUT_FILE = "model.bin"

NUMERICAL = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL = ["species", "island"]


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sex"] = df["sex"].replace(["."], np.nan)
    df = df.dropna(subset=["sex"]).copy()

    df.columns = df.columns.str.replace(" ", "_").str.lower()
    for col in CATEGORICAL + ["sex"]:
        df[col] = df[col].astype(str).str.replace(" ", "_").str.lower()

    df["sex"] = (df["sex"] == "male").astype(int)
    return df.reset_index(drop=True)


def train():
    print(f"Loading data from {DATA_PATH}...")
    df = load_and_clean_data(DATA_PATH)

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train["sex"].values
    y_test = df_test["sex"].values

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(df_full_train[CATEGORICAL + NUMERICAL].to_dict(orient="records"))
    X_test = dv.transform(df_test[CATEGORICAL + NUMERICAL].to_dict(orient="records"))

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=2,
        random_state=1,
        n_jobs=-1,
    )
    model.fit(X_full_train, y_full_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"Test ROC AUC: {auc:.4f}")

    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model saved successfully to {OUTPUT_FILE}")


if __name__ == "__main__":
    train()
