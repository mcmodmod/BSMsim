# BSMsim

An interactive Streamlit application for simulating asset price dynamics using Geometric Brownian Motion (GBM) and pricing European options with the Black–Scholes–Merton (BSM) model.

## Usage
As of now, the application can only be run locally.
1. Clone the repository:
```
git clone https://github.com/mcmodmod/BSMsim.git
cd BSMsim/
```
2. In a virtual environment, install dependencies with the package installer of your choice, e.g. with pip:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3. Run the app:
```
streamlit run main.py
```

**Please note:** The app contains a Monte-Carlo simulation. Large simulation sizes may require significant memory and computation time.
