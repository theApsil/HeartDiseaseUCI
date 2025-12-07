import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ---------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# ---------------------------------------------------------

df = pd.read_csv("data/heart_disease_uci.csv")
df = df.replace({"TRUE": 1, "FALSE": 0})

df = df.drop("id", axis=1)


y = df["num"]
X = df.drop("num", axis=1)


categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns


# ---------------------------------------------------------
# 2. PREPROCESSOR PIPELINE (SHARED)
# ---------------------------------------------------------

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ]
)


# ---------------------------------------------------------
# 3. DEFINE MODELS
# ---------------------------------------------------------

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs"
    ),

    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=len(y.unique()),
        learning_rate=0.05,
        max_depth=5,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )
}


# ---------------------------------------------------------
# 4. TRAIN / EVALUATE ALL MODELS
# ---------------------------------------------------------

results = []

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for model_name, model_obj in models.items():

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", model_obj)
    ])

    print(f"Training model: {model_name}")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "F1 Macro": f1_macro,
        "F1 Weighted": f1_weighted
    })

# ---------------------------------------------------------
# 5. OUTPUT RESULTS AS TABLE
# ---------------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\n\n========= MODEL COMPARISON =========\n")
print(results_df)

best_model = results_df.iloc[0]
print("\nBest model:", best_model["Model"])
print("Accuracy:", best_model["Accuracy"])
