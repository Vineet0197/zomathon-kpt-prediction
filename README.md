# Zomathon 2026 - Problem Statement 1
## Smart KPT Prediction Enhancement System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

### Problem
**Kitchen Prep Time (KPT)** prediction inaccuracies cause:
- Riders waiting at restaurants (wasted time & earnings)
- Cold food deliveries (poor customer experience)
- Inaccurate ETAs (customer dissatisfaction)

Root cause: **Food Order Ready (FOR)** signals are unreliable - merchants often mark food "ready" only when riders arrive.

---

## Our Solution

A dual-component approach combining **hardware** (IoT) and **software** (algorithms):

| Component | Description |
|-----------|-------------|
| **Smart Pickup Hub** | IoT device with dual QR scan capturing ground-truth timestamps |
| **Signal Reliability Framework** | Software algorithm scoring merchant reliability & fusing multiple signals |

### Key Results (50,000 Order Simulation)

| Metric | Current | Enhanced | Improvement |
|--------|---------|----------|-------------|
| Avg Rider Wait | ~5.2 min | ~2.1 min | **↓ 59%** |
| P50 Prediction Error | ~4.8 min | ~1.2 min | **↓ 75%** |
| On-Time Pickup Rate | ~52% | ~87% | **↑ 67%** |

---

## Quick Start

### Installation

```bash
git clone https://github.com/[your-username]/zomathon-kpt-prediction.git
cd zomathon-kpt-prediction
pip install -r requirements.txt
```

### Usage

#### Option 1: Generate Synthetic Data (Demo/Testing)
```bash
python kpt_analysis_visualization.py --output-dir ./results
```

#### Option 2: Use Your Own Production Data 🔑
```bash
python kpt_analysis_visualization.py \
    --data-file /path/to/your/orders.csv \
    --merchants-file /path/to/your/merchants.csv \
    --output-dir ./results
```

---

## 📊 Using Your Own Data (For Zomato Reviewers)

The script is designed to work with **real Zomato production data**. Simply provide your CSV files matching the expected schema:

### Required: `orders.csv`

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `order_id` | string | Unique order identifier | ✅ |
| `merchant_id` | string | Merchant/restaurant identifier | ✅ |
| `order_time` | datetime | Order placement timestamp | ✅ |
| `merchant_for_time` | datetime | Merchant FOR marking timestamp | ✅ |
| `rider_arrival` | datetime | Rider arrival at merchant | ✅ |
| `rider_pickup` | datetime | Rider pickup timestamp | ✅ |
| `actual_food_ready` | datetime | Actual food ready time (ground truth) | Optional |
| `actual_kpt_minutes` | float | Actual kitchen prep time | Optional |
| `merchant_tier` | string | Tier1/Tier2/Tier3 classification | Optional |
| `order_complexity` | int | 1-5 complexity scale | Optional |

**Column name variations supported**: The script auto-detects common variations like:
- `orderid`, `order_number` → `order_id`
- `restaurant_id`, `store_id` → `merchant_id`
- `order_timestamp`, `created_at` → `order_time`
- `pickup_time`, `picked_up_at` → `rider_pickup`

### Optional: `merchants.csv`

| Column | Type | Description |
|--------|------|-------------|
| `merchant_id` | string | Merchant identifier |
| `tier` | string | Volume tier classification |
| `avg_daily_orders` | int | Average daily order count |

> **Note**: If merchants file is not provided, merchant profiles are automatically derived from order data.

### What the Script Does With Your Data

1. **Validates & Preprocesses**: Auto-detects column names, parses dates, derives missing fields
2. **Calculates Merchant Reliability**: Applies our Signal Reliability Framework formula
3. **Simulates Enhanced System**: Uses Weighted Signal Fusion to predict improved KPT
4. **Generates Visualizations**: Creates 7 comprehensive charts
5. **Outputs Summary**: JSON file with key metrics and improvements

---

## Key Algorithms

### 1. Signal Reliability Framework

Scores each merchant's FOR signal reliability (0-1) based on:

```
Reliability = 1 - (0.40 × Bias_Ratio + 0.35 × Wait_Score + 0.25 × KPT_CV)
```

| Factor | Weight | Description |
|--------|--------|-------------|
| Bias Ratio | 40% | % of FOR signals within 60s of rider arrival (gaming indicator) |
| Wait Score | 35% | Normalized average rider wait time |
| KPT CV | 25% | Coefficient of variation in prep times (unpredictability) |

### 2. Weighted Signal Fusion Algorithm

Combines multiple KPT signals with adaptive weights:

| Scenario | Signal Weights |
|----------|---------------|
| Has IoT Device | Device: 70%, Rider-Inferred: 30% |
| High Reliability (≥0.7) | Merchant FOR: 55%, Rider: 45% |
| Medium Reliability | Merchant FOR: 35%, Rider: 65% |
| Low Reliability (<0.4) | Merchant FOR: 20%, Rider: 80% |

**Rider-Inferred KPT** = `max(rider_pickup, rider_arrival) - order_time`
- This signal is bias-resistant because it doesn't depend on merchant's FOR marking

---

## Output Files

After running, your output directory will contain:

| File | Description |
|------|-------------|
| `orders_processed.csv` | Input data with derived fields |
| `merchants_processed.csv` | Merchant profiles with reliability scores |
| `analysis_summary.json` | Key metrics and improvements |
| `viz_01_eda_overview.png` | Order distribution & wait time analysis |
| `viz_02_for_bias_analysis.png` | FOR bias patterns by tier |
| `viz_03_kitchen_load_impact.png` | Peak hour & complexity impact |
| `viz_04_comparison_dashboard.png` | Before/after comparison |
| `viz_05_reliability_impact.png` | Reliability score distribution |
| `viz_06_architecture.png` | Solution architecture diagram |
| `viz_07_business_impact.png` | ROI and deployment cost analysis |

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: DATA CAPTURE                        │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   IoT Device    │   Rider App     │    Merchant App             │
│   (Tier 1 & 2)  │   • QR Scanner  │    • FOR Marking            │
│   • Dual Scan   │   • GPS         │    • Rush Toggle            │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         v                 v                       v
┌─────────────────────────────────────────────────────────────────┐
│                LAYER 2: SIGNAL RELIABILITY FRAMEWORK            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Signal         │  Reliability    │   Kitchen Load              │
│  Cleaning       │  Scoring        │   Features (KBI)            │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         └─────────────────┼───────────────────────┘
                           v
┌─────────────────────────────────────────────────────────────────┐
│                LAYER 3: PREDICTION ENGINE                       │
├─────────────────────────────────────────────────────────────────┤
│              Weighted Signal Fusion Algorithm                   │
│              → Enhanced KPT Prediction → ETA Engine            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Business Impact

### Deployment Cost
| Tier | Merchants | Cost/Device | Total |
|------|-----------|-------------|-------|
| Tier 1 | 30,000 | ₹2,000 | ₹6 Cr |
| Tier 2 | 90,000 | ₹700 | ₹6.3 Cr |
| Tier 3 | 180,000 | ₹75 (sticker) | ₹1.35 Cr |
| **Total** | **300,000** | - | **₹13.65 Cr** |

### ROI
- **Daily Savings**: ₹15.4 Cr (reduced rider wait costs + customer retention)
- **Breakeven**: ~0.9 days
- **First Year ROI**: >1000%

---

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
```

---

## Team

**Team [Your Team Name]**
- [Member 1] - [Role]
- [Member 2] - [Role]

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Zomato Engineering Team for the problem statement
- OWASP for secure coding guidelines referenced in implementation
