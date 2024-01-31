# Web Tests
Web tests are selenium-based browser interaction tests, that fully simulate
actual user's behaviours.

# Preparation
- Have Google Chrome (Any version) installed.
- Install following python packages with `pip`:
    - `selenium`
    - `webdriver-manager`

# Run Tests
- Have WebUI with ControlNet installed running on `localhost:7860`
- Run `python main.py --overwrite_expectation` for the first run to set a 
baseline.
- Run `python main.py` later to verify the baseline still holds.