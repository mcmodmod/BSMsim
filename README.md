# BSMsim

An interactive Streamlit application for simulating asset price dynamics using Geometric Brownian Motion (GBM) and pricing European options with the Black–Scholes–Merton (BSM) model.

## Usage
To use the app, open [BSMsim](https://bsmsim.streamlit.app/) in a browser.

To run the application locally, follow these steps:
1. Clone the repository:
```
git clone https://github.com/mcmodmod/BSMsim.git
cd BSMsim/
```
2. In a virtual environment, install dependencies with the package installer of your choice, e.g. with pip:
```
pip install -r requirements.txt
```
3. Run the app:
```
streamlit run main.py
```

**Please note:** The app contains a Monte-Carlo simulation. Large simulation sizes may require significant memory and computation time. The number of simulations is capped at 2,000,000 to avoid crashing the streamlit servers. If you want to exceed this limit when running the app locally, you will have to change the `MAX_MC_SIMS` variable in `main.py`.

## Future Improvements
 - Add Greeks
 - Performance optimisation for large-scale simulations
 - Support for American options
