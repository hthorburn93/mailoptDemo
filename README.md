# Mailopt Demonstration package
Code to build a sorting centre network, build associated model, and solve. Also includes data of a toy example network. Results of this code shown in Thorburn, et. al, 2023 (See bottom for full citation).

Contains code to build a network, associated model and solve.
Contains data on the toy example network.
DOES NOT contain data on the full mail centre examples from the paper (due to commercial sensitivity).

## To install
Requires:
* Python (version 3.9-3.13) - see https://www.python.org/downloads/ for installation advice
* Poetry - see https://python-poetry.org/docs/ for installation advice

To install
1. Download/clone the github repository
2. In a terminal, navigate to the folder containing the pyproject.toml file
3. Run `poetry install` in the terminal

## Usage
To use the package, users need to be in a virtual environment. Poetry will automatically create one and install it there, but if you prefer, you can create one yourself, and install the package in there
If you want to use your own venv, you need to activate it, then run poetry install (as above)

If you want to use poetry's venv, you can simply run `poetry shell` to start the environment

For an example of how to use, please see the "mailoptDemo/scripts/Demo_Script.py" file. To run this file either
1. Run `poetry shell` to enter the venv, then run  `python scripts/Demo_Script.py`
2. Run `poetry run python scripts/Demo_Script.py` (this doesn't leave the venv change after running script)

## License

See License file

## Citation
Thorburn, H., Sachs, A., Fairbrother, J., Boylan, J.E., (2023) "A time-expanded network design model for staff allocation in mail centres" _IN REVIEW_
