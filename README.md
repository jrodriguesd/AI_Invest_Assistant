# AI_Invest_Assistant

This repository is based on the
[https://github.com/Damonlee92/Build_Your_Own_AI_Investor](https://github.com/Damonlee92/Build_Your_Own_AI_Investor)

Jupiter notebooks were converted to python
The functions were saved in utils.py

main.py is only the driver of the functions performed by the program

This project is a useful exercise to learn Machine Learning

## Building & Running Locally

1. Create a venv

```
conda create -n AI_Invest_Assistant python=3.12
```

2. Clone the project to any directory where you do development work

```
git clone https://github.com/jrodriguesd/AI_Invest_Assistant.git
```
3. Install the required libraries

```
pip install -r requirements.txt
```

4. rename .env_SAMPLE to .env
5. update SIMFIN_API_KEY with your simfin key [https://www.simfin.com](https://www.simfin.com). You need a simfin key to get the data needed by the program.

6. Run te program asking for help

```
python main.py -h
```

```
usage: main.py [-h] [-g] [-b] [-t] [-p] [-s] [-f]

options:
  -h, --help            show this help message and exit
  -g, --generate-data-file
                        Generate Data & Clean it
  -b, --back-test       backtest
  -t, --generate-test-data
                        Generate test data file
  -p, --plot-test-data  Generate test data file
  -s, --stock-picks     Print stock picks
  -f, --feature-importance
                        Plot feature importance
```
