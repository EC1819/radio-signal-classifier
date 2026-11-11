import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.logistic_scratch import ScratchLogisticRegression


def test_scratch_agrees_with_sklearn():
    df = pd.read_csv("data/features.csv")
    subset = df[df["mod"].isin(["BPSK", "QPSK"]) & (df["snr"] >= 0)]

    X = subset.drop(columns=["mod", "snr"]).to_numpy()
    y = (subset["mod"] == "QPSK").to_numpy().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scratch_model = ScratchLogisticRegression(learning_rate=0.5, n_iters=1000)
    scratch_model.fit(X_train_scaled, y_train)

    sklearn_model = LogisticRegression(penalty=None, max_iter=5000)
    sklearn_model.fit(X_train_scaled, y_train)

    scratch_preds = scratch_model.predict(X_test_scaled)
    sklearn_preds = sklearn_model.predict(X_test_scaled)

    agreement = (scratch_preds == sklearn_preds).mean()
    assert agreement >= 0.98
