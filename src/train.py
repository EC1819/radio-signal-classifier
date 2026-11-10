import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../data/features.csv")
X = df.drop(columns=["mod", "snr"])
y = df["mod"]
snrs = df["snr"]

strata = y + "_" + snrs.astype(str)

X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    X, y, snrs,
    test_size=0.2,
    stratify=strata,
    random_state=42,
)

strata_train = y_train + "_" + snr_train.astype(str)

X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train,
    train_size=20000,
    stratify=strata_train,
    random_state=42,
)

models = {
    "k-NN": (
        KNeighborsClassifier(),
        {"kneighborsclassifier__n_neighbors": [5, 15, 31, 61]},
    ),
    "Logistic Regression (L2)": (
        LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000),
        {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Logistic Regression (L1, saga)": (
        LogisticRegression(penalty="l1", solver="saga", max_iter=5000),
        {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"decisiontreeclassifier__max_depth": [5, 10, 20, None]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "randomforestclassifier__n_estimators": [100, 200],
            "randomforestclassifier__max_depth": [10, 20, None],
        },
    ),
}

results = {}

for name, (model, param_grid) in models.items():
    print(f"\nTraining {name}")
    pipeline = make_pipeline(StandardScaler(), model)

    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_tune, y_tune)

    best = search.best_index_
    results[name] = {
        "mean_accuracy": search.best_score_,
        "std": search.cv_results_["std_test_score"][best],
        "best_params": search.best_params_,
    }
    print(f"{search.best_score_:.3f} ± {results[name]['std']:.3f}  {search.best_params_}")

results_df = pd.DataFrame(results).T.sort_values("mean_accuracy", ascending=False)
results_df.to_csv("../figures/cv_results.csv")
print(results_df)
