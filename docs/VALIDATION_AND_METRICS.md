# Validation Strategy and Evaluation Metrics

**Document Version:** 1.0.0 (Phase 0)  
**Author:** AI/ML Engineering Pair  
**Project:** Smart Traffic Forecasting and Congestion Analytics  
**Status:** Pre-Implementation Specification

---

## 1. Time-Series Validation Philosophy

In standard tabular machine learning (e.g., image classification, house price prediction), observations are frequently assumed to be Independent and Identically Distributed (i.i.d.). Under this assumption, random $K$-fold cross-validation or random train-test splitting is standard practice.

**For time-series forecasting, random splitting is a fatal methodological error.**

### 1.1 Why Random Shuffling Causes Catastrophic Leakage
1. **Autocorrelation and Temporal Proximity:** Traffic measurements separated by 15 minutes are strongly correlated ($y_t \approx y_{t+1}$). If random shuffling places $t$ in the training set and $t+1$ in the test set, the model simply memorizes the immediate neighbor rather than learning underlying temporal dynamics.
2. **Lookahead Bias:** Shuffling allows the model to observe data from the future (e.g., a Tuesday afternoon) to predict the past (e.g., Tuesday morning), an impossibility in real-world deployment.

```
WRONG (Random Split):
[Train t=1] [Test t=2] [Train t=3] [Test t=4] [Train t=5]  <-- Future leaks into Past!

CORRECT (Chronological Split):
|==================== TRAIN (70%) ====================|====== VAL (15%) ======|====== TEST (15%) =====|
t=1                                                   t=k                     t=m                     t=N
(Learn patterns & fit scalers)                        (Model selection)       (Final unseen evaluation)
```

### 1.2 Splitting Protocol
- **Chronological Partitions:**
  - **Training Set (first 70%):** Used for fitting model parameters and feature transformation pipelines.
  - **Validation Set (subsequent 15%):** Used for hyperparameter tuning, lag window selection, and model comparison.
  - **Test Set (final 15%):** Held out as strictly unseen data for final reporting and generalizability auditing.
- **Multi-Sensor Synchronization:** For Dataset A (36 sensors), splitting is performed strictly along the **time axis**. All 36 sensor readings observed at timestamp $T$ reside in the identical split. Never place some sensors at time $T$ in training and others at time $T$ in testing.
- **Strict Preprocessing Isolation:** Scalers (`StandardScaler`, `MinMaxScaler`), mean imputers, and target encodings are fitted **exclusively on the training split** and applied transformationally to validation and test splits.

---

## 2. Leakage Prevention Rules in Feature Engineering

| Feature Category | Permitted Construction | Forbidden Construction (Leakage Bug) |
| :--- | :--- | :--- |
| **Lag Features** | $y_{t-1}, y_{t-2}, y_{t-12}$ (Shift index $\ge +1$) | $y_{t+1}, y_t$ as inputs when predicting $y_{t+1}$ |
| **Rolling Means / Std** | `df.rolling(window=W, closed='left')` or `df.shift(1).rolling(W).mean()` | `df.rolling(window=W, center=True).mean()` (centers window on future steps) |
| **Target Encodings** | Mean traffic calculated over training split only | Mean traffic calculated across full dataset |
| **Normalization** | `scaler.fit(X_train); scaler.transform(X_test)` | `scaler.fit_transform(full_X)` |

---

## 3. Planned Model Hierarchy

We adopt a disciplined, CPU-friendly model hierarchy starting from an uncompromising baseline to sophisticated tabular tree ensembles:

### 3.1 Model 1: Naive Persistence Baseline
$$\hat{y}_{t+1} = y_t$$
- **Description:** Assumes the traffic measurement 15 minutes into the future equals the current measurement.
- **Significance:** In short-horizon traffic forecasting, persistence is surprisingly difficult to beat because traffic states evolve gradually. Any proposed ML algorithm that fails to consistently outperform persistence across MAE and RMSE is useless in practice.

### 3.2 Model 2: Ridge Regression
$$\min_{w} \|y - Xw\|_2^2 + \alpha \|w\|_2^2$$
- **Description:** Regularized linear regression applying $L_2$ shrinkage on feature coefficients.
- **Benefits:** Fast, deterministic, mathematically transparent, and virtually instantaneous on CPU. Serves as our linear machine learning benchmark.

### 3.3 Model 3: Random Forest Regressor
- **Description:** Non-linear ensemble of de-correlated decision trees trained with bootstrap aggregation.
- **Benefits:** Captures non-linear relationships, multi-sensor interactions, and thresholding effects without requiring feature normalization.

### 3.4 Model 4: Histogram-Based Gradient Boosting (`HistGradientBoostingRegressor`)
- **Description:** Scikit-learn’s optimized gradient boosting implementation inspired by LightGBM.
- **Benefits:**
  - Bins continuous numerical features into integer histograms (256 bins), dramatically reducing memory bandwidth and training time.
  - Native multi-threaded CPU parallelization.
  - Native support for missing values without synthetic imputation.
  - Exceptional accuracy on tabular time-series lag structures.

### 3.5 Deep Learning Policy
We explicitly **defer deep recurrent architectures (LSTM, GRU) or Temporal Fusion Transformers (TFT)** until simpler baselines are thoroughly audited. Neural networks introduce extensive compute overhead, hyperparameter volatility, and risk of overfitting on non-stationary series. Deep learning will only be explored if tree ensembles demonstrate clear residual deficiencies that justify the added complexity.

---

## 4. Evaluation Metrics Formulation

To provide an honest, multifaceted assessment, we evaluate models across accuracy, operational speed, and peak performance:

### 4.1 Primary Accuracy Metrics

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   - *Interpretation:* Average magnitude of forecast error in physical units (e.g., vehicles/hr or occupancy). Robust to extreme outliers.

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   - *Interpretation:* Strongly penalizes large, sudden forecasting errors. Essential for transportation operators where large mispredictions cause scheduling chaos.

3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   - *Interpretation:* Proportion of variance explained by the model relative to the historical mean. Interpreted cautiously as time-series mean references can be misleading.

4. **Relative Improvement Over Baseline ($\Delta\%$):**
   $$\Delta\%_{\text{MAE}} = \frac{\text{MAE}_{\text{persistence}} - \text{MAE}_{\text{model}}}{\text{MAE}_{\text{persistence}}} \times 100\%$$
   - *Interpretation:* The definitive benchmark metric demonstrating whether the machine learning pipeline adds genuine value over the trivial persistence heuristic.

### 4.2 Multi-Sensor Spatial Breakdown (Dataset A)
Because Dataset A features 36 distinct sensors, global aggregate metrics can mask localized failures. We will report:
- Per-sensor MAE and RMSE: $\text{MAE}_s \quad \forall s \in \{1, 2, \dots, 36\}$
- Summary distribution across sensors: Minimum, 25th percentile, Median, 75th percentile, and Maximum sensor error.

### 4.3 Computational & System Metrics
- **Training Time:** Elapsed wall-clock time (seconds) to fit model on the training split.
- **Inference Latency:** Average prediction time per 1,000 samples (milliseconds) on CPU.
- **Serialized Model Size:** Disk footprint (`.joblib` artifact size in MB).
- **Peak Memory Usage:** Maximum RAM utilized during feature generation and training (MB).

### 4.4 High-Traffic / Peak-Period Metrics
Forecasting accuracy during empty roads (midnight to 4:00 AM) is trivial and can artificially inflate global metrics. We will calculate a separate **Peak-Period MAE**:
- Measured strictly on observations falling into the top $10\%$ traffic volume percentile or predefined diurnal peak commute hours (e.g., 07:30–09:30 and 16:30–18:30).

---

## 5. Metric Reporting Integrity Policy

> [!IMPORTANT]
> **No Fabricated Results Guarantee**  
> In Phase 0, all performance metrics, leaderboard rankings, and error values are strictly unpopulated placeholders. No numbers will be reported until actual model training and evaluation are conducted on verified data in Phase 3.
