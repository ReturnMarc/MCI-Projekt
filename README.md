# MCI-Projekt

## Environment
Use environment.yml file for setting up your enviornment. Run the following commands when your current directory is the project folder.

```cmd
conda env create -f environment.yml
conda activate xai-dashboard
```

## Project Structure
```
MCI-Projekt/
├── main.py          # Main application entry point
├── layout.py        # Dashboard layout definition
├── callbacks.py     # Callback functions
├── model_utils.py   # Model training and XAI functions
├── model_store.py   # Global model storage
├── environment.yml  # Conda environment file
└── housing.csv      # Dataset
```

## Usage
Start the dashboard by running:
```cmd
python main.py
```