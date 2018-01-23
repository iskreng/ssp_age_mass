# ssp_age_mass
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/iskreng/ssp_age_mass/master?filepath=SSP_color_color.ipynb)

Calculate BC03 SSP mass from F606W, F814W and F160W HST/WFC3 magnitudes. It uses the maximum likelihood value using a normal distribution model for the magnitude and its error.

Usage:
1. An example with default values for a target distance modulus, and magnitudes that generates a plot in which the calculated age, M/L and Mass are labeled.

python3 SSP_color_color.py 

2. Custom magnitudes and distance modulus:
python3 SSP_color_color.py --m606 <value> --m814 <value> --m160 <value> --mM <value>
