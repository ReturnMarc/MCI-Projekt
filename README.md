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
├── main.py # Main application entry point 
├── layout.py # Dashboard layout definition 
├── callbacks.py # Callback functions 
├── model_utils.py # Model training and XAI functions 
│
├── assets/ # Static assets 
│ └── style.css # Dashboard styling 
│
├── datasets/ # Dataset directory 
│ ├── diabetes.csv 
│ ├── housing.csv 
│ ├── retail_sales_dataset.csv 
│ └── ...
│
├── stored_models/ # Trained model storage 
│ ├── diabetes_models.joblib 
│ ├── housing_models.joblib
│ ├── retail_sales_dataset_models.joblib 
│ └── ...
│
└── environment.yml # Conda environment file
```

## Usage
Start the dashboard by running:
```cmd
python main.py
```
During the first startup, all the models for the datasets in the subdirectory "datasets" are trained. They will only be trained once and then stored under "stored models". So the next time you start up the dashboard none or only newly added datasets have to train their corresponding models. 

If everything is sucessful, your console will look like this after the first start. You can click on the provided link for the Dash-App.

```cmd
Training models for new dataset: diabetes.csv
Training models for new dataset: retail_sales_dataset.csv
Training models for new dataset: StudentPerformanceFactors.csv
Training models for new dataset: titanic.csv
Training models for new dataset: WineQT.csv
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'main'
 * Debug mode: on
No new datasets to train - all datasets have existing models
```
