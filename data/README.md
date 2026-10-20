# Data Directory Governance and Structure

This directory houses the raw and processed datasets for the **Smart Traffic Forecasting and Congestion Analytics** project.

## Directory Layout

```
data/
├── raw/                                        <- Immutable original raw files downloaded from official sources.
│   ├── dataset_a_uci_734/
│   │   ├── traffic_dataset.mat                 <- Verified MATLAB v7 matrix archive (36 sensors, 2101 timesteps).
│   │   ├── Traffic Flow Prediction Dataset.docx <- Author's official documentation.
│   │   └── traffic_flow_forecasting-1.zip      <- Downloaded zip archive.
│   └── dataset_b_uci_492/
│       ├── Metro_Interstate_Traffic_Volume.csv <- Extracted 48,204 row CSV file.
│       ├── Metro_Interstate_Traffic_Volume.csv.gz
│       └── metro_interstate_traffic_volume.zip
├── processed/                                  <- Cleaned, validated, chronologically-split artifacts ready for ML.
└── README.md                                   <- This governance document.
```

## Phase 1 Verified Findings & Governance

### 1. Dataset A: UCI 734 (Traffic Flow Forecasting)
- **Status:** Downloaded and verified.
- **File Format:** MATLAB `.mat` containing sparse CSC matrices (`scipy.sparse._csc.csc_array`) in cell arrays.
- **Content:**
  - `tra_X_tr` ($1 \times 1261$ cell array; each cell is a $36 \times 48$ sparse matrix).
  - `tra_Y_tr` ($36 \times 1261$ dense float64 matrix; target normalized in $[0, 1]$).
  - `tra_X_te` ($1 \times 840$ cell array; each cell is a $36 \times 48$ sparse matrix).
  - `tra_Y_te` ($36 \times 840$ dense float64 matrix; target normalized in $[0, 1]$).
  - `tra_adj_mat` ($36 \times 36$ binary symmetric adjacency matrix; 86 directed edges).
- **Temporal Alignment:** The test set begins at step 1261 with zero gap, perfectly continuing the sliding window from step 1260. Time series represents 2101 contiguous 15-minute intervals across weekdays.
- **Defensible Chronological Validation:** **CONFIRMED**. Row order $t$ strictly indexes continuous time.

### 2. Dataset B: UCI 492 (Metro Interstate Traffic Volume)
- **Status:** Downloaded, decompressed, and verified.
- **File Format:** CSV (48,204 rows, 9 columns).
- **Target Variable:** `traffic_volume` (integer hourly vehicle counts in $[0, 7280]$).
- **Timestamps:** 2012-10-02 09:00:00 to 2018-09-30 23:00:00. Chronologically sorted (`is_monotonic_increasing == True`).
- **Duplicate Timestamps:** 7,629 duplicate timestamp rows (15.83%). Audited: `traffic_volume` is 100% identical across all duplicate timestamp groups; duplicates exist solely due to concurrent weather descriptions (e.g. "Rain" and "Drizzle").
- **Deduplication Strategy for Phase 2:** Aggregate or select primary weather event per hourly timestamp before generating lag features.

### 3. Strict Dataset Isolation
Datasets A and B originate from different states (Virginia/DC vs. Minnesota), different time scales (15-min vs. 1-hour), and different target representations (normalized $[0, 1]$ vs. raw vehicle counts). They are stored in strictly segregated subdirectories and will never be combined.
