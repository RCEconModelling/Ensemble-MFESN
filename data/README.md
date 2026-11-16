# Data access

The GDP inputs originate from the research project for *Reservoir Computing for
Macroeconomic Forecasting with Mixed Frequency Data (2024)*:

[Upstream research repository](https://github.com/RCEconModelling/Reservoir-Computing-for-Macroeconomic-Modelling)

## Included benchmark predictions

This repository retains eight DFM benchmark prediction CSVs:

```text
data/
  dfm_medium/
    stock__10__6__[1]_fixed__2007/
      train_predictions_12.csv
      test_predictions_12.csv
    almon__10__6__[1]_fixed__2007/
      train_predictions_12.csv
      test_predictions_12.csv
  dfm_small/
    stock__5__6__[1]_fixed__2007/
      train_predictions_12.csv
      test_predictions_12.csv
    almon__5__6__[1]_fixed__2007/
      train_predictions_12.csv
      test_predictions_12.csv
```

These are supplied benchmark predictions, not raw GDP predictor panels. The
Medium-MD and Small-MD B=1 scripts read them during full estimation; they do not
estimate the DFM benchmarks themselves.

## Original predictor data are not distributed

The original GDP predictor data cannot be uploaded or redistributed because of
licensing restrictions, including those on daily series sourced from Datastream.
Installing this repository does not download those data or grant access to them.
Users wishing to refit the GDP models must obtain the inputs separately under
the applicable access and licensing conditions.

The scripts expect prepared inputs named:

```text
data/preprocessed/fullsample_output_data.csv
data/preprocessed/fullsample_output_filled_data.csv
```

Neither prepared file is distributed. Keep the original preprocessing, date
indices, variable definitions and sample periods; arbitrary replacement data
will not reproduce the reported results. B=100 and the frozen-grid same-budget
exercise require the first CSV, including the extended pre-evaluation sample.
B=1 estimation additionally uses the filled CSV and the relevant DFM predictions.

The saved-result table and figure workflows do not require either prepared CSV.
The simulations generate their own data or losses. See the
[detailed reproduction guide](../REPRODUCING.md) for the table/figure map, commands and
exercise-specific requirements.
