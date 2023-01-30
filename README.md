# bikeshare-experiments

## Requirements

Requirements:
- Python 3
- Java
- Maven
- Pip
- Python packages in requirements.txt
- Online-PSL

### Python Packages
Python package requirement are provided in requirements.txt.
```
pip3 install -r scripts/requirements.txt
```

### Online-PSL
This project requires online-PSL to run (https://github.com/linqs/online-psl).
Ensure both Maven and Java are installed. Then clone the Online-PSL repository and execute "mvn clean install" at its root.

```
git clone https://github.com/linqs/online-psl
mvn clean install
```

## Data
To load the PSL formatted data used in the paper run the `data/fetchData.sh` script from the data directory.

```
cd data
./fetchData.sh
```

### Data Construction
To reconstruct the PSL formatted data from the raw datasets first modify the `DATASET` variable in 
`scripts/data_construction/construct.py` **and** `scripts/data_construction/predicate_constructors.py`
 to select the dataset you plan to run. Modify the `DATA_URL`, `DATA_FILE`, and `EXTRACTED_DATA_FILE` to 
 the appropriate values for the raw dataset. Then run:

```
python3 scripts/data_construction/construct.py
```

## Experiments
To run the experiments, execute the script.
```
./run.sh
```
This will run all the experiments reported in the paper.
Experiment options can be modified in `scrips/run_experiments.sh`

### Citation
If you find this repository useful please cite the following paper:

```
@inproceedings{dickens:hicss23,
    title = {Online Collective Demand Forecasting for Bike Sharing Services},
    author = {Charles Dickens and Alex Miller and Lise Getoor},
    booktitle = {Hawaii International Conference on System Sciences},
    year = {2023},
    _publisher = {HICSS},
    address = {Maui},
}
```
