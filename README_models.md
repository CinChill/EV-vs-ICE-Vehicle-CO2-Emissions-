# EV vs ICE Model Training

Bu klasordeki `cleaned_EV_vs_ICE_vehicle_specs.csv` dosyasini kullanarak uc model egitir:

- `vehicle_category_model.joblib`: teknik ozelliklerden `Vehicle_Category` tahmini
- `co2_regression_model.joblib`: teknik ozelliklerden `CO2_Emissions_g_per_mile` tahmini
- `environmental_impact_model.joblib`: CO2 degerinden turetilen cevresel etki seviyesi tahmini

## Kurulum

```powershell
pip install -r requirements.txt
```

## Egitimi calistirma

```powershell
python train_vehicle_models.py --data cleaned_EV_vs_ICE_vehicle_specs.csv --outdir model_outputs
```

Egitimden sonra `model_outputs` klasorunde modeller, metrikler, karisiklik matrisleri, ozellik onemleri ve cevresel etki ozetleri olusur.

## Siteyi calistirma

```powershell
python app.py
```

Tarayicida `http://127.0.0.1:5000` adresini ac. Site `model_outputs` klasorundeki egitilmis modelleri kullanir.

Farkli model klasoru kullanmak istersen:

```powershell
$env:MODEL_DIR="model_outputs_no_fuel"
python app.py
```

## Daha siki deney

`Fuel_Type`, EV/ICE sinifini cok guclu ele verdigi icin daha zor bir deney yapmak istersen:

```powershell
python train_vehicle_models.py --drop-fuel-type --outdir model_outputs_no_fuel
```

Marka ezberini azaltmak icin `--drop-make` de ekleyebilirsin.
