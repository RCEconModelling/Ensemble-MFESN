# From Many Models, One: Macroeconomic Forecasting with Reservoir Ensembles

### Giovanni Ballarin<sup>1</sup>, Lyudmila Grigoryeva<sup>1,2</sup> and Yui Ching Li<sup>3</sup>

<sup>1</sup>University of St. Gallen. 
<br>
<sup>2</sup>Honorary Associate Professor, University of Warwick.
<br>
<sup>3</sup>Bank for International Settlements - The views expressed in this work are solely those of the authors and do not necessarily reflect those of the Bank for International Settlements.

#### Project structure:

* `data`: data loaded to construct forecasts. 

* `figures_multistep`: all exported figures used in the paper. 

* `results`: pickled (`.pkl`) files for all numerical experiments in the paper. 

* `python`: model code; multi-frequency ESN model libraries; script with estimation, forecasting and simulations for the Medium-MD dataset.

    - Please check `requirements.txt` before running.

    - Run `ensemble_multistep_medium_md.py` to run forecast combinations and build tables/plots.

    - Flag `DO_FORECASTS = False` is required, as the full dataset cannot currently be shared due to licensing restrictions. Forecasts will be loaded from disk.

    - Set `DO_PLOTS = True` to also generate main plots.

    - Set `OPTIONAL_PLOTS = True` to generate optional plots.

    - Modify `PATH_PREFIX` as needed on your machine.

For details regarding data copyrights and sources, please refer to `data/README.md`.

## Acknowledgements

YCL and LG acknowledge the financial support of the FoKo of the University of St. Gallen (Project Nr.1022324, _Machine Learning Techniques for Macroeconomic Forecasting and Pricing of Financial Derivatives_). GB thanks the Great Minds Postdoctoral Fellowship program of the University of St. Gallen, which made this research possible.

<!-- ## Reference 
```bibtex
@article{ballarinFromManyModelsOne2025,},
}
```
-->