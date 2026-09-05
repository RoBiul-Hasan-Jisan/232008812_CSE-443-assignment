
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

RANDOM_STATE = 42
os.makedirs("../models", exist_ok=True)
os.makedirs("../reports", exist_ok=True)
os.makedirs("../reports/figures", exist_ok=True)




#  Load data

df = pd.read_csv("../data/bank-full.csv", sep=";")
print("Shape:", df.shape)
print(df.head())




# Exploratory Data Analysis

print("\n Missing values ")
print(df.isnull().sum())

print("\n Target distribution ")
print(df["y"].value_counts())
print(df["y"].value_counts(normalize=True))

target_counts = df["y"].value_counts()
plt.figure(figsize=(5, 4))
sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title("Target class distribution (y)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../reports/figures/target_distribution.png", dpi=150)
plt.close()

numeric_cols_raw = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()
for i, col in enumerate(numeric_cols_raw):
    sns.boxplot(data=df, x="y", y=col, ax=axes[i])
    axes[i].set_title(col)
for j in range(len(numeric_cols_raw), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("../reports/figures/numeric_by_target.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols_raw].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation among numeric features")
plt.tight_layout()
plt.savefig("../reports/figures/correlation_heatmap.png", dpi=150)
plt.close()

categorical_cols_raw = ["job", "marital", "education", "default", "housing",
                         "loan", "contact", "month", "poutcome"]
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()
for i, col in enumerate(categorical_cols_raw):
    rate = df.groupby(col)["y"].apply(lambda s: (s == "yes").mean()).sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, ax=axes[i])
    axes[i].set_title(f"Subscription rate by {col}")
    axes[i].tick_params(axis="x", rotation=45)
    axes[i].set_ylabel("Subscription rate")
plt.tight_layout()
plt.savefig("../reports/figures/categorical_subscription_rates.png", dpi=150)
plt.close()


#  age distribution, overall and split by outcome
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df["age"], bins=30, kde=True, ax=axes[0], color="#4c72b0")
axes[0].set_title("Age distribution (all customers)")
axes[0].set_xlabel("age")
sns.kdeplot(data=df, x="age", hue="y", common_norm=False, fill=True,
            alpha=0.35, ax=axes[1], palette={"no": "#4c72b0", "yes": "#dd8452"})
axes[1].set_title("Age distribution by subscription outcome (normalized)")
axes[1].set_xlabel("age")
plt.tight_layout()
plt.savefig("../reports/figures/age_distribution.png", dpi=150)
plt.close()

#  subscription rate by age group (binned), with group size
age_bins = [17, 25, 35, 45, 55, 65, 100]
age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
age_group_stats = df.groupby("age_group", observed=True).agg(
    subscription_rate=("y", lambda s: (s == "yes").mean()),
    n_customers=("y", "size"),
)
fig, ax1 = plt.subplots(figsize=(9, 5.5))
bars = ax1.bar(age_group_stats.index.astype(str), age_group_stats["subscription_rate"],
               color="#55a868")
ax1.set_ylabel("Subscription rate")
ax1.set_xlabel("Age group")
ax1.set_title("Subscription rate by age group (bar height) with group size (labels)")
for bar, n in zip(bars, age_group_stats["n_customers"]):
    ax1.annotate(f"n={n:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("../reports/figures/subscription_rate_by_age_group.png", dpi=150)
plt.close()

#  monthly call volume vs. subscription rate (dual-axis)
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
monthly = df.groupby("month").agg(
    calls=("y", "size"),
    subscription_rate=("y", lambda s: (s == "yes").mean()),
).reindex(MONTH_ORDER).dropna(how="all")
fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax1.bar(monthly.index, monthly["calls"], color="#8c9eb2", label="Calls made")
ax1.set_ylabel("Number of calls made", color="#4c72b0")
ax1.set_xlabel("Month")
ax1.tick_params(axis="y", labelcolor="#4c72b0")
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly["subscription_rate"], color="#c44e52",
         marker="o", linewidth=2, label="Subscription rate")
ax2.set_ylabel("Subscription rate", color="#c44e52")
ax2.tick_params(axis="y", labelcolor="#c44e52")
plt.title("Call volume vs. subscription rate by month")
fig.tight_layout()
plt.savefig("../reports/figures/monthly_volume_vs_rate.png", dpi=150)
plt.close()

#  "unknown" category frequency by column
cat_cols_with_unknown = ["job", "education", "contact", "poutcome"]
unknown_counts = {col: (df[col] == "unknown").sum() for col in cat_cols_with_unknown}
unknown_pct = {col: (df[col] == "unknown").mean() * 100 for col in cat_cols_with_unknown}
plt.figure(figsize=(7.5, 5))
bars = plt.bar(unknown_counts.keys(), unknown_counts.values(), color="#937860")
plt.ylabel("Count of 'unknown' entries")
plt.title("'Unknown' category frequency by column")
for bar, col in zip(bars, unknown_counts):
    plt.annotate(f"{unknown_counts[col]:,}\n({unknown_pct[col]:.1f}%)",
                 (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("../reports/figures/unknown_values_by_column.png", dpi=150)
plt.close()


#  Data leakage note: 'duration'

print("\nRows with duration == 0:", (df["duration"] == 0).sum())
print("Of those, y == 'no' in all cases:", (df.loc[df["duration"] == 0, "y"] == "no").all())




#  Preprocessing

df_model = df.copy()
df_model["y"] = (df_model["y"] == "yes").astype(int)


df_model["was_previously_contacted"] = (df_model["pdays"] != -1).astype(int)
df_model["pdays_clean"] = df_model["pdays"].replace(-1, 0)

numeric_features = ["age", "balance", "day", "campaign", "pdays_clean", "previous"]
categorical_features = ["job", "marital", "education", "default", "housing",
                         "loan", "contact", "month", "poutcome"]
engineered_features = ["was_previously_contacted"]

feature_cols_no_duration = numeric_features + categorical_features + engineered_features
feature_cols_with_duration = feature_cols_no_duration + ["duration"]

X_no_dur = df_model[feature_cols_no_duration]
X_with_dur = df_model[feature_cols_with_duration]
y = df_model["y"]




#  Train/test split (stratified, since target is imbalanced ~88/12)

X_train_nd, X_test_nd, y_train, y_test = train_test_split(
    X_no_dur, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
X_train_wd, X_test_wd, _, _ = train_test_split(
    X_with_dur, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("\nTrain size:", X_train_nd.shape, " Test size:", X_test_nd.shape)
print("Train target balance:\n", y_train.value_counts(normalize=True))
print("Test target balance:\n", y_test.value_counts(normalize=True))


def make_pipeline(numeric_cols, categorical_cols, engineered_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
            ("eng", "passthrough", engineered_cols),
        ]
    )
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
    return pipe





#  Hyperparameter tuning (regularization strength) via grid search

param_grid = {"classifier__C": [0.01, 0.1, 1, 10, 100]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def tune_and_fit(X_train, y_train, numeric_cols, categorical_cols, engineered_cols, label):
    pipe = make_pipeline(numeric_cols, categorical_cols, engineered_cols)
    grid = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"\n[{label}] Best C: {grid.best_params_}, Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_score_


model_no_dur, cv_auc_no_dur = tune_and_fit(
    X_train_nd, y_train, numeric_features, categorical_features, engineered_features,
    "No duration (deployable model)"
)
model_with_dur, cv_auc_with_dur = tune_and_fit(
    X_train_wd, y_train, numeric_features + ["duration"], categorical_features, engineered_features,
    "With duration (upper-bound reference model)"
)




#  Find the optimal decision threshold (F1-maximizing) on a validation


X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
    X_train_nd, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
)


threshold_search_model = clone(model_no_dur)
threshold_search_model.fit(X_tr_sub, y_tr_sub)
y_val_proba = threshold_search_model.predict_proba(X_val)[:, 1]

prec_val, rec_val, thresholds_val = precision_recall_curve(y_val, y_val_proba)
f1_val_scores = 2 * (prec_val * rec_val) / (prec_val + rec_val + 1e-12)
best_threshold = float(thresholds_val[np.argmax(f1_val_scores[:-1])])
print(f"\nBest F1-maximizing threshold (from validation split): {best_threshold:.3f}")




# Profit-optimal threshold (distinct from the F1-optimal one above)


cost_per_call = 1
profit_per_subscription = 50

candidate_thresholds = np.linspace(0.01, 0.99, 99)
val_profits = []
for t in candidate_thresholds:
    pred_t = (y_val_proba >= t).astype(int)
    tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_val, pred_t).ravel()
    calls_v = tp_v + fp_v
    profit_v = (tp_v * profit_per_subscription) - (calls_v * cost_per_call)
    val_profits.append(profit_v)

profit_optimal_threshold = float(candidate_thresholds[int(np.argmax(val_profits))])
print(f"Profit-optimal threshold (from validation split, "
      f"${profit_per_subscription}/sub vs ${cost_per_call}/call): "
      f"{profit_optimal_threshold:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(candidate_thresholds, val_profits)
plt.axvline(best_threshold, color="green", linestyle="--", label=f"F1-optimal ({best_threshold:.3f})")
plt.axvline(profit_optimal_threshold, color="red", linestyle="--",
            label=f"Profit-optimal ({profit_optimal_threshold:.3f})")
plt.xlabel("Decision threshold")
plt.ylabel(f"Validation-set net profit (${profit_per_subscription}/sub, ${cost_per_call}/call)")
plt.title("Profit vs. threshold (validation split)")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/figures/profit_vs_threshold.png", dpi=150)
plt.close()





#  Evaluation

def evaluate(model, X_test, y_test, label, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    print(f"\n=== {label} — Test set performance (threshold={threshold:.3f}) ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}" if isinstance(v, float) else f"{k:>10}: {v}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"]))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    return metrics, y_pred, y_proba, cm



metrics_nd_default, pred_nd_default, proba_nd, cm_nd_default = evaluate(
    model_no_dur, X_test_nd, y_test, "No duration (default threshold)", threshold=0.5
)
# F1-tuned threshold
metrics_nd, pred_nd, _, cm_nd = evaluate(
    model_no_dur, X_test_nd, y_test, "No duration (F1-tuned threshold)", threshold=best_threshold
)

metrics_nd_profit, pred_nd_profit, _, cm_nd_profit = evaluate(
    model_no_dur, X_test_nd, y_test, "No duration (profit-optimal threshold)",
    threshold=profit_optimal_threshold
)

metrics_wd, pred_wd, proba_wd, cm_wd = evaluate(
    model_with_dur, X_test_wd, y_test, "With duration (reference model)", threshold=0.5
)




#  Visual diagnostics

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, cm, title in zip(
    axes, [cm_nd_default, cm_nd, cm_nd_profit, cm_wd],
    ["No duration (0.5 threshold)", f"No duration (F1={best_threshold:.3f})",
     f"No duration (profit={profit_optimal_threshold:.3f})", "With duration"]
):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    ax.set_title(f"Confusion Matrix\n({title})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("../reports/figures/confusion_matrices.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 6))
for proba, label in [(proba_nd, "No duration"), (proba_wd, "With duration")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/figures/roc_curves.png", dpi=150)
plt.close()

# Precision-Recall curve (more informative than ROC under class imbalance)
plt.figure(figsize=(7, 6))
prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba_nd)
pr_auc = average_precision_score(y_test, proba_nd)
plt.plot(rec_curve, prec_curve, label=f"No duration (PR-AUC={pr_auc:.3f})")
plt.axhline(y_test.mean(), color="gray", linestyle="--", label="Baseline (prevalence)")
plt.scatter(
    [recall_score(y_test, pred_nd)], [precision_score(y_test, pred_nd, zero_division=0)],
    color="red", zorder=5, label=f"Tuned threshold ({best_threshold:.3f})"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (No-duration model)")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/figures/pr_curve.png", dpi=150)
plt.close()


prob_true, prob_pred = calibration_curve(y_test, proba_nd, n_bins=10)
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="No-duration model")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of actual positives")
plt.title("Calibration Curve (No-duration model)")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/figures/calibration_curve.png", dpi=150)
plt.close()




#  Feature importance (coefficients) — deployable (no-duration) model

preprocessor = model_no_dur.named_steps["preprocessor"]
clf = model_no_dur.named_steps["classifier"]

num_names = numeric_features
cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))
eng_names = engineered_features
all_feature_names = num_names + cat_names + eng_names

coef_df = pd.DataFrame({
    "feature": all_feature_names,
    "coefficient": clf.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)

print("\nTop 15 features by |coefficient| (No-duration model):")
print(coef_df.head(15).to_string(index=False))
coef_df.to_csv("../reports/feature_coefficients.csv", index=False)

plt.figure(figsize=(9, 8))
top20 = coef_df.head(20).sort_values("coefficient")
colors = ["#d62728" if c < 0 else "#2ca02c" for c in top20["coefficient"]]
plt.barh(top20["feature"], top20["coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Top 20 Logistic Regression Coefficients (No-duration model)")
plt.xlabel("Coefficient (standardized features / one-hot categories)")
plt.tight_layout()
plt.savefig("../reports/figures/top_coefficients.png", dpi=150)
plt.close()




#  Business profit/cost analysis


def profit_summary(pred, label):
    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, pred).ravel()
    calls_ = tp_ + fp_
    gross_ = tp_ * profit_per_subscription
    cost_ = calls_ * cost_per_call
    net_ = gross_ - cost_
    print(f"{label:38s} | calls={calls_:5d} | TP={tp_:4d} | "
          f"gross=${gross_:>7,} | cost=${cost_:>6,} | net=${net_:>7,}")
    return {"calls": int(calls_), "true_positives": int(tp_),
            "gross_profit": int(gross_), "total_cost": int(cost_), "net_profit": int(net_)}


baseline_pred = np.ones(len(y_test), dtype=int)  # call everyone

print(f"\n Profit analysis (${profit_per_subscription}/subscription, ${cost_per_call}/call) ")
profit_default = profit_summary(pred_nd_default, "Default threshold (0.5)")
profit_f1 = profit_summary(pred_nd, f"F1-optimal threshold ({best_threshold:.3f})")
profit_optimal = profit_summary(pred_nd_profit, f"Profit-optimal threshold ({profit_optimal_threshold:.3f})")
profit_baseline = profit_summary(baseline_pred, "Baseline (call everyone)")

net_profit = profit_optimal["net_profit"]
baseline_net_profit = profit_baseline["net_profit"]
print(f"\nProfit-optimal strategy vs. calling everyone: "
      f"${net_profit - baseline_net_profit:,} difference")
print(f"Profit-optimal strategy vs. F1-optimal strategy: "
      f"${net_profit - profit_f1['net_profit']:,} difference")




#  Error analysis — look at missed subscribers (false negatives)


X_test_copy = X_test_nd.copy()
X_test_copy["y_true"] = y_test.values
X_test_copy["y_pred"] = pred_nd_profit
missed_subscribers = X_test_copy[(X_test_copy["y_true"] == 1) & (X_test_copy["y_pred"] == 0)]
caught_subscribers = X_test_copy[(X_test_copy["y_true"] == 1) & (X_test_copy["y_pred"] == 1)]

print(f"\n Error analysis: missed subscribers (false negatives) ")
print(f"Missed {len(missed_subscribers)} of {int(y_test.sum())} actual subscribers "
      f"({len(missed_subscribers) / y_test.sum():.1%})")
print("\nProfile of MISSED subscribers (numeric features):")
print(missed_subscribers[numeric_features].describe().loc[["mean", "50%"]])
print("\nProfile of CORRECTLY CAUGHT subscribers (numeric features):")
print(caught_subscribers[numeric_features].describe().loc[["mean", "50%"]])

missed_subscribers.to_csv("../reports/missed_subscribers.csv", index=False)




#  Save the trained deployable model and thresholds to disk

joblib.dump(model_no_dur, "../models/logreg_deployable_model.pkl")
with open("../reports/best_threshold.txt", "w") as f:
    f.write(json.dumps({
        "f1_optimal_threshold": best_threshold,
        "profit_optimal_threshold": profit_optimal_threshold,
        "recommended_for_deployment": "profit_optimal_threshold",
    }, indent=2))
print("\ndin")
print("din")




# Example: Predicting on one unseen customer


example_customer = pd.DataFrame([{
    "age": 35,
    "balance": 1500,
    "day": 10,
    "campaign": 1,
    "pdays_clean": 0,                 
    "previous": 0,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "poutcome": "unknown",
    "was_previously_contacted": 0
}])[feature_cols_no_duration]

# Predict probability of subscription
example_proba = model_no_dur.predict_proba(example_customer)[0, 1]

# Decisions using different thresholds
default_prediction = example_proba >= 0.50
profit_prediction = example_proba >= profit_optimal_threshold

# Display results
print("\nExample Prediction: Unseen Customer")
print(example_customer.to_string(index=False))

print(f"\nPredicted probability of subscribing (y=1): {example_proba:.4f}")
print(f"Decision @ default threshold (0.50): {'Call (Yes)' if default_prediction else 'Skip (No)'}")
print(
    f"Decision @ profit-optimal threshold ({profit_optimal_threshold:.3f}): "
    f"{'Call (Yes)' if profit_prediction else 'Skip (No)'}"
)