# EV vs ICE Vehicle Analysis and Prediction

This project analyzes technical specifications of electric vehicles (EV) and internal combustion engine vehicles (ICE), prepares a cleaned dataset, trains machine learning models, and serves the trained models through a local Flask web dashboard.

## Project Overview

The project focuses on two main goals:

- Predicting the vehicle category from technical specifications.
- Estimating environmental impact using CO2 emissions and vehicle efficiency features.

The workflow includes preprocessing, exploratory data analysis, model training, evaluation, and a web interface for interactive predictions.

## Dataset

The project uses vehicle specification data from 2015 to 2026.

Main files:

- `EV_vs_ICE_Vehicle_Specs_2015_2026.csv`: original dataset
- `cleaned_EV_vs_ICE_vehicle_specs.csv`: cleaned dataset used for training
- `Preprocessing.ipynb`: preprocessing and exploratory analysis notebook
- `EV_vs_ICE_Preprocessing_EDA_Report.pdf`: PDF report of preprocessing and EDA

Important columns include:

- `Make`
- `Year`
- `Fuel_Type`
- `Engine_Cylinders`
- `Engine_Size_L`
- `Drivetrain`
- `City_MPG`
- `Highway_MPG`
- `Combined_MPG`
- `CO2_Emissions_g_per_mile`
- `EV_Range_miles`
- `Vehicle_Category`
- `Transmission_Type`

## Preprocessing Summary

The preprocessing notebook performs the following steps:

- Loads and inspects the original dataset.
- Checks dataset shape, column types, missing values, and duplicates.
- Fills missing `Fuel_Type` values with `Unknown`.
- Drops the high-cardinality `Model` column.
- Simplifies raw transmission values into `Automatic`, `Manual`, and `Other`.
- Reviews categorical and numeric feature distributions.
- Creates visual EDA charts for vehicle category, year, fuel type, CO2 emissions, MPG, EV range, and correlations.
- Saves the final cleaned dataset as `cleaned_EV_vs_ICE_vehicle_specs.csv`.

## Models

The training script builds three models:

- Vehicle category classifier: predicts `Vehicle_Category`.
- CO2 regression model: predicts `CO2_Emissions_g_per_mile`.
- Environmental impact classifier: predicts impact level derived from CO2 emissions.

Model outputs are saved under `model_outputs/`, including:

- `.joblib` trained model files
- classification reports
- feature importance tables
- confusion matrix images
- CO2 regression evaluation outputs
- environmental impact summaries

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- joblib
- Flask
- HTML, CSS, JavaScript

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Models

Run the training script:

```bash
python train_vehicle_models.py --data cleaned_EV_vs_ICE_vehicle_specs.csv --outdir model_outputs
```

For a stricter experiment that removes the highly informative fuel type feature:

```bash
python train_vehicle_models.py --drop-fuel-type --outdir model_outputs_no_fuel
```

You can also remove brand information to reduce make-based memorization:

```bash
python train_vehicle_models.py --drop-fuel-type --drop-make --outdir model_outputs_no_fuel_no_make
```

## Run the Web Dashboard

After training the models, start the Flask app:

```bash
python app.py
```

Open the dashboard in your browser:

```text
http://127.0.0.1:5000
```

The dashboard lets you enter vehicle technical specifications and returns:

- predicted vehicle category
- predicted CO2 emissions
- environmental impact level
- annual CO2 estimate
- five-year CO2 estimate
- class probability charts

## Use a Different Model Directory

By default, the app reads models from `model_outputs/`.

To use a different trained model directory:

```powershell
$env:MODEL_DIR="model_outputs_no_fuel"
python app.py
```

On macOS/Linux:

```bash
MODEL_DIR=model_outputs_no_fuel python app.py
```

## Project Structure

```text
.
+-- app.py
+-- train_vehicle_models.py
+-- requirements.txt
+-- README.md
+-- README_models.md
+-- Preprocessing.ipynb
+-- EV_vs_ICE_Preprocessing_EDA_Report.pdf
+-- EV_vs_ICE_Vehicle_Specs_2015_2026.csv
+-- cleaned_EV_vs_ICE_vehicle_specs.csv
+-- templates/
|   +-- index.html
+-- static/
|   +-- app.js
|   +-- styles.css
+-- model_outputs/
    +-- vehicle_category_model.joblib
    +-- co2_regression_model.joblib
    +-- environmental_impact_model.joblib
```

## Notes

- The dataset is imbalanced: most rows are `ICE (Gas)`, while some categories have very few samples.
- The training script skips labels with too few rows for reliable classification.
- `Fuel_Type` is a very strong predictor of EV vs ICE category, so experiments without this feature are useful for a more challenging evaluation.
- `.joblib` files are configured for Git LFS through `.gitattributes`.

## License

This project is prepared for educational and data mining purposes.
