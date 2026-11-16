# From Many Models, One: Macroeconomic Forecasting with Reservoir Ensembles

### Giovanni Ballarin<sup>1</sup>, Lyudmila Grigoryeva<sup>1,2</sup> and Yui Ching Li<sup>3</sup>

<sup>1</sup>University of St. Gallen. 
<br>
<sup>2</sup>Honorary Associate Professor, University of Warwick.
<br>
<sup>3</sup>Bank for International Settlements - The views expressed in this work are solely those of the authors and do not necessarily reflect those of the Bank for International Settlements.

Code and saved results for the paper's GDP forecasting exercises and
forecast-combination simulations.

**[Detailed reproduction guide: tables, figures, commands and required data](REPRODUCING.md)**

## Project structure

- `python/`: forecasting, forecast-combination and simulation scripts.
- `results*_rev1/`: revised saved forecasts, losses, seeds and exercise notes.
- [tables_rev1/](tables_rev1/): manuscript table exports.
- [figures_rev1/](figures_rev1/): manuscript figure panels.
- [data/](data/): supplied DFM benchmark predictions and data-access notes.
- `results/` and `figures_multistep/`: historical public-release outputs.

## Using the code

The supplied saved results allow the manuscript tables and figures to be
reproduced without refitting the GDP models or obtaining the original predictor
panel. See [REPRODUCING.md](REPRODUCING.md) for the script-to-table/figure map,
commands, full-regeneration options and numerical limitations.
Dependencies are listed in [pyproject.toml](pyproject.toml).

## Data access

The original GDP predictor data are not distributed because of licensing
restrictions. Full GDP re-estimation requires those inputs to be obtained
separately; the simulation exercises generate their own data or losses.
See the [data-access notes](data/README.md).

## Acknowledgements

YCL and LG acknowledge the financial support of the FoKo of the University of St. Gallen (Project Nr.1022324, _Machine Learning Techniques for Macroeconomic Forecasting and Pricing of Financial Derivatives_). GB thanks the Great Minds Postdoctoral Fellowship program of the University of St. Gallen, which made this research possible.

<!-- ## Reference 
```bibtex
@article{ballarinFromManyModelsOne2025,},
}
```
-->
