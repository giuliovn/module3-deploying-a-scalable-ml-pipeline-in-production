GitHub link: https://github.com/giuliovn/module3-deploying-a-scalable-ml-pipeline-in-production

### Run training
```commandline
pip install -r requirements.txt
dvc pull data/census_clean.csv
python train/train_model.py data/census_clean.csv
```
#### Outputs
Model and performance files will be written to `model/` directory

### Run tests
#### Training
```commandline
pytest -v tests/model
```
### API
pytest -v tests/api/api.py
