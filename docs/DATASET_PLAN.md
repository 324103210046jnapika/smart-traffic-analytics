# Dataset Plan & Verified Schema Audit

**Document Version:** 1.2.0 (Phase 2 Verified)  
**Author:** AI/ML Engineering Pair  
**Project:** Smart Traffic Forecasting and Congestion Analytics  
**Status:** Chronological Partitioning & Feature Pipeline Completed (Awaiting Approval for Phase 3)

---

## 1. Overview & Verification Summary

During Phase 1 and Phase 2, both official datasets were acquired from the UC Irvine Machine Learning Repository and subjected to exhaustive data forensics, chronological partitioning, and causal feature engineering.

| Dimension | Dataset A (Main Multi-Sensor) | Dataset B (Weather-Aware Benchmark) |
| :--- | :--- | :--- |
| **Repository ID** | UCI #734 (`traffic+flow+forecasting-1`) | UCI #492 (`metro+interstate+traffic+volume`) |
| **Citation** | Zhao et al. (ACM TSAS 2019) | John Hogue (UCI ML Repository 2019) |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Physical Scope** | 36 highway sensors in Northern VA / Washington D.C. | Westbound I-94 in Minneapolis-St. Paul, Minnesota |
| **Raw Format** | MATLAB v7 `.mat` (sparse CSC matrices in cell arrays) | Compressed CSV (`.csv.gz` / `.csv`) |
| **Total Observations** | 2,101 contiguous 15-minute steps | 48,204 rows (40,575 unique hourly timestamps) |
| **Time Range** | 2017-01-02 00:00 to 2017-02-01 00:15 (Weekdays) | 2012-10-02 09:00:00 to 2018-09-30 23:00:00 (6 years) |
| **Nominal Horizon** | "15 minutes ahead" (Author high-level prose) | 1 hour ahead |
| **Verified Physical Horizon** | **2 quarter-hours (30 minutes)** ahead of latest input lag | **1 hour ahead** |
| **Target Variable** | `tra_Y` (Normalized traffic flow in $[0.0, 1.0]$) | `traffic_volume` (Raw hourly vehicle counts $[0, 7280]$) |
| **Chronological Audit** | **VERIFIED CONTIGUOUS** (Step $t$ strictly indexes time) | **VERIFIED CHRONOLOGICAL** (Monotonically increasing) |
| **Missing Values** | Zero NaNs across all cells | 0 NaNs in target/weather (48,143 NaNs in `holiday`) |
| **Spatial Graph** | Provided as $36 \times 36$ adjacency matrix `tra_adj_mat` | Single corridor (No spatial graph needed) |

---

## 2. Dataset A: Verified Forecast Horizon and Anti-Leakage Audit

### 2.1 The Forecast Horizon Discrepancy Clarified
- **Published Text Statement:** The introductory Word document states: *"The goal is to predict the traffic volume 15 minutes into the future for all sensor locations."*
- **Empirical Mathematical Reality:** We systematically tested the offset between target $Y[:, t]$ and the 10 input lag features across all 1,259 available time transitions:
  $$\text{At Offset } 0: \quad \text{Full vector match percentage} = 0.0\%$$
  $$\text{At Offset } 1: \quad \text{Full vector match percentage} = 0.0\%$$
  $$\text{At Offset } 2: \quad \text{Full vector match percentage} = 100.0\% \quad (1259 / 1259)$$
- **Physical Interpretation:**
  - In each input sample $X[t]$, `lag[9]` represents the latest observed traffic measurement at forecast time $k$.
  - At step $t+1$ (15 minutes later), the window advances by 1 position.
  - At step $t+2$ (30 minutes later), the target observation $Y[:, t]$ enters the window at `lag[9]`.
  - Therefore, relative to the latest available input feature (`lag[9]`), the target $Y[:, t]$ is **strictly 2 quarter-hours (30 minutes) ahead**.
- **Engineering Decision:** We retain the official author target array `tra_Y` untouched so our models remain strictly comparable with published benchmarks, but we formally document that the effective physical forecast horizon is **30 minutes ahead of the latest input measurement**.

### 2.3 Option A Split Implementation
Dataset A is partitioned strictly along the time axis under **Option A**:
- **Train Split:** Steps 0 to 999 (1,000 steps, $47.6\%$) $\to 36,000$ tabular rows in `train.parquet`.
- **Validation Split:** Steps 1,000 to 1,260 (261 steps, $12.4\%$) $\to 9,396$ tabular rows in `val.parquet`.
- **Test Split:** Steps 1,261 to 2,100 (840 steps, $40.0\%$) $\to 30,240$ tabular rows in `test.parquet`.
- **Untouched Benchmark Guarantee:** The official 840 test steps remain strictly held out and will not be inspected or evaluated until final testing in Phase 5.

---

## 3. Dataset B: Causal Cleaning, Deduplication & Rainfall Rationale

### 3.1 Duplicate Weather Aggregation Rationale
- **Observation:** 13,074 rows share an hourly timestamp across 5,445 unique timestamps.
- **Forensic Finding:** In **100% of duplicate cases**, `traffic_volume` is identical (0 variance). Duplicates occur solely because OpenWeatherMap recorded multiple simultaneous weather phenomena (e.g., both Rain and Drizzle at 07:00).
- **Zero Lookahead Justification:** All duplicate records share the identical timestamp $t$. Aggregating records that occur at the exact same hour $t$ utilizes **zero future information**.
- **Aggregation Rule:** Group by `date_time`:
  - `traffic_volume`: `first` (exact match across all duplicates).
  - Numerical weather features (`temp`, `rain_1h`, `snow_1h`, `clouds_all`): contemporaneous `median`.
  - Categorical weather (`weather_main`): highest meteorological severity occurring in that hour (Squall > Thunderstorm > Snow > Rain > Drizzle > Fog/Mist/Haze > Clouds > Clear).

### 3.2 Causal Temperature Repair (Preventing Temporal Leakage)
- **Observation:** Exactly 10 rows in Jan/Feb 2014 logged `temp = 0.0 K` ($-273.15^\circ$C) due to temporary sensor telemetry dropouts. Surrounding winter temperatures were $255.93 \text{ K}$ ($-17.2^\circ$C).
- **Leakage Prevention Rule:** We **strictly reject two-sided linear interpolation** because two-sided interpolation peeks forward in time to interpolate backward into the past.
- **Causal Solution:** We apply **strictly causal forward-fill (`.ffill()`)** from the preceding valid winter reading. During an outage, the system carries forward the last known physical reading. This utilizes strictly past observations and introduces zero future information.

### 3.3 Rainfall Anomaly Rationale (rain_1h == 9831.3 mm)
- **Observation:** Exactly 1 row on `2016-07-11 17:00:00` logged `rain_1h = 9831.3 mm` (9.8 meters of water in 1 hour).
- **Domain Rationale:** The global 1-hour rainfall world record is 305 mm; the Minnesota state record is 132 mm. An hourly reading of 9,831 mm is an unphysical sensor transmission error.
- **Empirical Ceiling:** Across all remaining 48,203 rows in the 6-year record, the maximum valid non-anomalous rainfall is **55.63 mm** (99.9th percentile is 13.46 mm).
- **Capping Strategy:** Values exceeding 100 mm are capped to the verified non-anomalous empirical ceiling of **55.63 mm**. This eliminates extreme leverage distortions in regression models while preserving time series continuity.
- **Raw Data Preservation:** The original raw CSV in `data/raw/dataset_b_uci_492/` remains completely unmodified and preserves the original `9831.3 mm` and `0.0 K` records.
