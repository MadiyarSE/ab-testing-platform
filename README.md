# A/B Testing Platform

Synthetic A/B testing platform demonstrating variance-reduction and 
causal inference techniques on realistic fintech transaction data.

## Status: In progress
- [x] Data generator (FX fee reduction experiment)
- [x] Stats engine (t-test, linearization, CUPED, stratification)
- [x] dbt models
- [ ] Airflow orchestration
- [ ] Dashboard

## Experiment 1: FX Fee Reduction

**Hypothesis:** Reducing currency exchange fee from 0.5% to 0.2% increases 
revenue per user via higher exchange frequency.

**Design:**
- 50/50 split via double hashing (bucket assignment + group assignment, separate salts)
- Stratified by country (UK, DE, PL, ES)
- CUPED covariate: historical FX revenue (4 weeks pre-experiment)

**Result:** Effect is statistically significant but negative 
(delta ~= -1.02 EUR/user, p<0.0001) -- the fee cut outweighs the 
frequency lift, meaning the change would reduce total FX revenue.

**Note on linearization:** standard linearization assumes a stable 
Y/X ratio across groups. Since fee itself differs by group here (it's 
the treatment), CUPED and stratification are more reliable than raw 
linearization for this specific metric design.

## Stack
- Python (pandas, numpy, scipy)
- SQLite (data storage)
- Statistical methods: t-test, linearization, CUPED, stratification

## Structure\
ab-testing-platform/
├── data_generator/
│ └── generate_fx_experiment.py # synthetic data generation + double hashing split
├── analyze_fx_experiment.py # stats engine: naive/linearization/CUPED/stratified tests
└── ab_platform.db # generated locally (gitignored, run generator to recreate)/

## How to run
```bash
pip install pandas numpy scipy
python3 data_generator/generate_fx_experiment.py
python3 analyze_fx_experiment.py
```
