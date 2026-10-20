# Smart Traffic Forecasting and Congestion Analytics — Final Portfolio Report

**Author / Candidate:** 2nd-Year BTech Student in Computer Science & Artificial Intelligence  
**Role:** Senior Machine Learning Engineer, Data Scientist, and Software Architect  
**Project Track:** Capstone Portfolio Project for ExcelR AI/ML Internship  
**System Status:** Complete Production Implementation (Phases 0 through 5)  
**Last Verified:** October 19, 2026  

---

## 1. Executive Summary

This project delivers an end-to-end, scientifically defensible machine learning platform for historical road network traffic forecasting and congestion analytics. Designed and executed to commercial engineering standards, the system:
- Ingests and processes two complementary open-source benchmarks from the **UCI Machine Learning Repository**:
  - **Dataset A (UCI #734):** Spatiotemporal highway traffic flow across a 36-sensor network in Silicon Valley, California.
  - **Dataset B (UCI #492):** 6-year hourly traffic volume and meteorological observations on Interstate 94 in Minneapolis-St. Paul, Minnesota.
- Implements strict, leakage-proof **chronological partitioning** with zero lookahead bias.
- Corrects a critical real-world telemetry flaw in Dataset B by identifying and excluding **2,589 invalid gap-bridging targets**, ensuring every modeled horizon is exactly 1 hour.
- Formally documents an empirical discrepancy in Dataset A where author documentation claimed a 15-minute horizon, but empirical numerical synchronization proved an exact **30-minute horizon** relative to latest input features.
- Benchmarks candidate models (**Ridge Regression, Random Forest, HistGradientBoosting**) against a strict **Naive Persistence Baseline** ($\hat{y}_{t+h} = y_t$).
- Validates model generalization on strictly held-out **Official Test Partitions**:
  - **Dataset A Official Test Set (30,240 samples):** HistGradientBoosting achieves **Test MAE = 0.0257** ($+30.96\%$ gain vs persistence, $R^2 = 0.9657$).
  - **Dataset B Official Test Set (5,675 hours):** Random Forest achieves **Test MAE = 143.71 veh/h** ($+75.74\%$ gain vs persistence, $R^2 = 0.9875$), closely followed by HistGradientBoosting at **Test MAE = 149.74 veh/h** ($+74.72\%$ gain, consuming only 376 KB on disk vs 60 MB for Random Forest).
- Serves an interactive multi-page **Streamlit 1.62 / Plotly 7.1** dashboard for EDA, model leaderboards, interactive residual forecasting, weather benchmarks, and traffic physics audits.
- Backed by an automated test suite of **33 unit tests** with 100% pass rate in 9.6 seconds.

> **Crucial Scientific Boundary:** This system models historical time series data. It is **not** a real-time GPS navigation tool or dynamic traffic control system. Furthermore, high traffic volume is strictly distinguished from physical congestion using macroscopic traffic flow physics ($q = k \cdot v$).

---

## 2. Hardware and Execution Environment

The entire development and evaluation pipeline was engineered for CPU-first execution on standard developer hardware:
- **Operating System:** Windows 11 (64-bit)
- **Runtime Environment:** Python 3.13.3 (x86-64)
- **Processor:** Intel Core i5 / Iris Xe integrated graphics (16 GB System RAM, shared video memory)
- **GPU Acceleration:** None assumed, required, or utilized.
- **Execution Strategy:** CPU-first vectorization via NumPy, SciPy, Pandas, and Scikit-Learn.
- **Reproducibility:** Seed 42 locked across all randomized algorithms. Total training time across all 4 models: **<4.5 seconds**.

---

## 3. Official Dataset Sources & Specifications

| Attribute | Dataset A (UCI #734) | Dataset B (UCI #492) |
|---|---|---|
| **Repository Title** | Traffic Flow Forecasting | Metro Interstate Traffic Volume |
| **Official Source** | [UCI Repository #734](https://archive.ics.uci.edu/dataset/734/traffic+flow+forecasting) | [UCI Repository #492](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) |
| **Geographic Domain** | Silicon Valley Freeways, CA | Interstate 94 (Eastbound), MN |
| **Spatial Network** | 36 Inductive Loop Sensors, 86 Directed Graph Edges | Single interstate highway segment |
| **Temporal Span** | 2,101 discrete 15-minute time steps | October 2012 to September 2018 (6 years) |
| **Target Variable** | Normalized traffic flow $[0.0, 1.0]$ | Hourly traffic volume (vehicles / hour) |
| **Verified Forecast Horizon** | **30 minutes ahead** ($t + 2$ steps beyond latest feature) | **1 hour ahead** ($t + 1\text{ hour}$, calendar verified) |
| **Exogenous Signals** | Sensor spatial topology (adjacency matrix) | Weather (temp, rain, snow, clouds) & Holidays |

---

## 4. Forensic Target Verification & Data Leakage Defenses

### 4.1. Dataset A Horizon Forensic Audit
- **Nominal Claim:** The author's `.docx` file accompanied by UCI 734 claims a *"15-minute ahead prediction"*.
- **Empirical Synchronization:** Each sample in input matrix $X$ contains 10 historical lag steps: $[t-9, t-8, \dots, t-1, t]$. Numerical inspection across all 36 sensors proved that target $Y[:, k]$ corresponds exactly to offset $+2$ steps beyond input feature `lag[9]` ($15\text{ min} \times 2 = 30\text{ min}$).
- **Literature Integrity:** Rather than altering the target or mislabeling the horizon, the official benchmark $Y$ was preserved, but formally documented as a **30-minute horizon** across all specifications, code, and visualizations.

### 4.2. Dataset B Causal Preprocessing & Leakage Prevention
- **Causal Deduplication:** 7,629 duplicate hourly timestamps were aggregated causally by retaining the latest chronological observation, avoiding lookahead averaging.
- **No Two-Sided Interpolation:** Missing weather values (10 rows of 0.0 K temperature) were filled using causal forward-fill (`.ffill()`). Future observations were never accessed.
- **Rolling Window Discipline:** All rolling features (`vol_roll_mean_3h`, `vol_roll_std_24h`) used `closed='left'` to ensure observation $t$ does not incorporate future signal.
- **Extreme Outlier Isolation:** Sensor rain glitch ($98,313\text{ mm}$) capped at the 6-year empirical maximum ($55.63\text{ mm}$).
- **Preprocessor Fitting:** `StandardScaler` and `OneHotEncoder` fitted exclusively on training rows.

---

## 5. Comprehensive Sample Accounting

A rigorous distinction is maintained between **warmup exclusions** (rows dropped to initialize historical lags) and **invalid timestamp-gap targets** (rows dropped because the succeeding observation was $> 1\text{ hour}$ away).

### Dataset A Sample Accounting
- **Total Steps in Benchmark:** 2,101 steps $\times$ 36 sensors = **75,636 samples**.
- **Warmup Exclusions:** **0 steps** (the official `.mat` structure pre-embedded 10 historical lags per step).
- **Training Partition (Option A Benchmark-Aligned):** Steps 0 to 999 (**1,000 steps** = 36,000 samples, 47.60%).
- **Validation Partition:** Steps 1,000 to 1,260 (**261 steps** = 9,396 samples, 12.42%).
- **Official Test Partition:** Steps 1,261 to 2,100 (**840 steps** = 30,240 samples, 39.98%).

### Dataset B Sample Accounting & Arithmetic Reconciliation
- **Raw CSV Records:** 48,204 rows.
- **Deduplicated Continuous Records:** 40,575 rows (7,629 duplicate timestamps aggregated causally).
- **Warmup Exclusions:** **168 rows** (first week required to initialize `vol_lag168` and `vol_roll_std_24h`).
- **Invalid Timestamp-Gap Targets:** **2,589 rows** ($\Delta t_{\text{next}} \neq 1\text{ hour}$ or final series endpoint).
- **Set Overlap & Arithmetic Reconciliation:**
  - Notice that 8 observations occurred within the initial 168-hour warmup window *and* were followed by an outage ($\Delta t_{\text{next}} > 1\text{ hour}$).
  - Observations dropped exclusively for warmup: $168 - 8 = \mathbf{160\text{ rows}}$.
  - Observations dropped exclusively for invalid gap targets: $2,589 - 8 = \mathbf{2,581\text{ rows}}$.
  - Observations dropped for both reasons simultaneously: $\mathbf{8\text{ rows}}$.
  - Total unique dropped rows: $|\text{Warmup} \cup \text{Invalid Gaps}| = 168 + 2,589 - 8 = \mathbf{2,749\text{ rows}}$.
- **Total Valid Modeled Pool:**
  $$40,575 - 2,749 = \mathbf{37,826\text{ samples}}$$

#### Exact Reconciled Partition Breakdown:
| Chronological Partition | Date / Timestamp Range | Raw Window Rows | Warmup-Only Exclusions | Invalid-Gap Exclusions | Dual Exclusions (Both) | Valid Modeled Samples |
|---|---|---|---|---|---|---|
| **Warmup Window** | 2012-10-02 09:00 to 2012-10-09 18:00 | 168 | 160 | 0 | 8 | **0** *(Warmup dropped)* |
| **Training (70%)** | 2012-10-09 19:00 to 2017-06-12 12:00 | 29,034 | 0 | 2,556 | 0 | **26,478** (69.99%) |
| **Validation (15%)** | 2017-06-12 13:00 to 2018-02-05 08:00 | 5,687 | 0 | 14 | 0 | **5,673** (15.00%) |
| **Official Test (15%)** | 2018-02-05 09:00 to 2018-09-30 22:00 | 5,686 | 0 | 11 *(10 gaps + 1 end)* | 0 | **5,675** (15.01%) |
| **Total** | **2012-10-02 to 2018-09-30** | **40,575** | **160** | **2,581** | **8** | **37,826** (100.0%) |

*Reconciliation Arithmetic:*
$$26,478\text{ (train)} + 5,673\text{ (val)} + 5,675\text{ (test)} = \mathbf{37,826\text{ valid samples}}$$
$$40,575\text{ (total)} - 160\text{ (warmup only)} - 2,581\text{ (gaps only)} - 8\text{ (both)} = \mathbf{37,826\text{ valid samples}}$$
*Boundaries:* Partitions are strictly non-overlapping and continuous ($\Delta t = \text{exactly } 1\text{ hour}$ between Train Max and Val Min, and between Val Max and Test Min).

---

## 6. Official Test Set Evaluation & Benchmarking Results

In accordance with Phase 5 protocol, models were frozen after validation selection and evaluated **once** on the official test partitions.

### 6.1. Dataset A: Multi-Sensor Network (30-Minute Ahead Horizon)
**Official Test Partition:** 840 time steps $\times$ 36 sensors = **30,240 samples**.

| Model | Test MAE | Test RMSE | Test $R^2$ | Gain vs Baseline | Val MAE | Generalization Gap (MAE) | Relative Change | Inference Latency |
|---|---|---|---|---|---|---|---|---|
| **Persistence Baseline** | 0.0372 | 0.0545 | 0.9288 | 0.00% | 0.0365 | +0.0007 | +1.9% | 0.00 $\mu$s |
| **Ridge Regression** | 0.0309 | 0.0455 | 0.9505 | +17.03% | 0.0298 | +0.0011 | +3.7% | 0.12 $\mu$s |
| **Random Forest** | 0.0264 | 0.0393 | 0.9630 | +29.13% | 0.0246 | +0.0018 | +7.3% | 4.06 $\mu$s |
| **HistGradientBoosting (Best)** | **0.0257** | **0.0379** | **0.9657** | **+30.96%** | **0.0241** | **+0.0016** | **+6.6%** | **52.87 $\mu$s** |

*Finding:* HistGradientBoosting confirms its superiority on unseen test data, delivering a **30.96% error reduction** over persistence with an exceptionally narrow generalization gap (+0.0016 MAE).

### 6.2. Dataset B: Interstate 94 Traffic Volume (1-Hour Ahead Horizon)
**Official Test Partition:** **5,675 gap-filtered hours**.

| Model | Test MAE (veh/h) | Test RMSE (veh/h) | Test $R^2$ | Gain vs Baseline | Val MAE | Generalization Gap (MAE) | Relative Change | Inference Latency |
|---|---|---|---|---|---|---|---|---|
| **Persistence Baseline** | 592.36 | 818.59 | 0.8298 | 0.00% | 587.68 | +4.68 | +0.8% | 0.00 $\mu$s |
| **Ridge Regression** | 362.00 | 486.05 | 0.9400 | +38.89% | 372.82 | -10.82 | -2.9% | 0.28 $\mu$s |
| **HistGradientBoosting** | 149.74 | 225.25 | 0.9871 | +74.72% | 162.92 | -13.18 | -8.1% | 5.73 $\mu$s |
| **Random Forest (Best)** | **143.71** | **221.65** | **0.9875** | **+75.74%** | **155.44** | **-11.73** | **-7.5%** | **9.37 $\mu$s** |

*Finding:* Both tree ensembles achieve over **74% improvement** over the persistence baseline on the official test set. The negative generalization gap (lower test error than validation error) reflects seasonal stability in the 2018 test period, confirming zero overfitting.

---

## 7. Macroscopic Traffic Flow Physics: Volume vs Congestion

A central contribution of this project is correcting the misconception that "high traffic volume = congestion".

### The LWR Traffic Flow Equation
In transportation engineering (Lighthill-Whitham-Richards model):
$$q = k \cdot v$$
Where $q$ is flow (veh/h), $k$ is density (veh/km), and $v$ is space-mean speed (km/h).

```
   Traffic Flow q (veh/h)
        ^
  q_max |          * * * (Capacity Flow)
        |       *         *
        |     *             * (Congested Regime: v low, k high)
        |    *               *
        |   * (Free-Flow)     *
        |  *                   * (Gridlock / Jam: q -> 0, v -> 0)
        +----------------------------------->
       0                   k_crit     k_jam   Density k (veh/km)
```

### The Congestion Duality Trap
Any flow rate $q < q_{\max}$ exists at two physical states:
1. **Free-Flow:** Low density, maximum speed (e.g. 1,000 veh/h at 3:00 AM moving at 100 km/h).
2. **Forced Gridlock:** Critical density, minimum speed (e.g. 1,000 veh/h during a 5:30 PM standstill moving at 10 km/h).

**Conclusion:** Traffic volume alone cannot mathematically classify congestion without speed or sensor occupancy telemetry. Our project explicitly models **traffic volume and flow**, avoiding unsubstantiated congestion claims.

---

## 8. Interactive Streamlit Application Architecture

The multi-page dashboard is organized modularly:
- **Landing Page (`streamlit_app.py`):** System KPIs, dual-dataset comparison matrix, model registry status, and scientific disclaimer.
- **Page 1 (`1_Overview_and_EDA.py`):** Interactive 36-sensor circular graph (86 directed edges), multi-sensor flow overlay, 24-hour diurnal rush hour curves, and sensor gap audit table.
- **Page 2 (`2_Model_Leaderboard.py`):** Toggle between Validation Leaderboard and Official Test Leaderboard, generalization gap bar charts, and per-sensor error distribution.
- **Page 3 (`3_Forecasting_Studio.py`):** Interactive sequence forecasting with dynamic window slider, dual-panel observed vs predicted series, real-time residual error bars, and 85th-percentile peak regime tracking.
- **Page 4 (`4_Weather_Benchmark.py`):** Analysis of adverse weather impact on Interstate 94, weather-segmented model error tables, and feature hierarchy discussion (88% variance from lags vs 12% marginal gain from weather).
- **Page 5 (`5_Risks_and_Limitations.py`):** Interactive Greenshields fundamental diagram curve, forensic horizon audit, lookahead prevention checklist, and commercial streaming architecture roadmap.

---

## 9. Automated Testing & Verification Audit

The system includes 33 automated unit tests executed via `pytest`:
- **Phase 0 (Setup & Config):** 4 tests (directory structure, documentation integrity, YAML validity, requirements).
- **Phase 1 (Data Ingestion & Isolation):** 4 tests (file existence, `.mat` schema, CSV schema, pipeline isolation).
- **Phase 2 (Splitting & Causal Features):** 7 tests (chronological boundaries, sensor sync, 30-min horizon verification, zero lookahead).
- **Phase 3 (Model Benchmarking):** 7 tests (metric precision, persistence logic, model fitting, finite predictions, per-sensor metrics, artifact serialization).
- **Phase 4 (Interactive Dashboard):** 6 tests (data loaders, model deserialization, batch inference consistency, Plotly figure generation, page syntax).
- **Phase 5 (Official Test Benchmarks):** 5 tests (test leaderboard files, sample count matching, generalization stability, per-sensor test consistency, test loaders).

**Audit Command:** `python -m pytest tests/ -v`  
**Result:** **33 passed in 9.60 seconds** (100% pass rate).

---

## 10. Reproduction Instructions

To reproduce the complete pipeline from scratch:
```powershell
# 1. Clone repository and navigate to root
cd C:\Users\Jnapikachowdary\.gemini\antigravity\scratch\smart-traffic-analytics

# 2. Rebuild chronological datasets (with gap exclusions)
python -m src.data.make_dataset

# 3. Train all models and benchmark on validation sets
python -m src.models.train_and_evaluate

# 4. Evaluate frozen models on official test partitions
python -m src.models.evaluate_test_sets

# 5. Execute complete automated test suite
python -m pytest tests/ -v

# 6. Launch the Streamlit interactive dashboard
python -m streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in any browser to explore the interactive platform.
