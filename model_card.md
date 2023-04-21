# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Classification model developed for learning purpose.

## Intended Use
The model predicts salary based on biographic and ethnic data.
It is intended only to use for demonstration purposes.

## Training Data
Raw data is provided by the UCI Machine Learning Repository.
Details can be found here: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
Evaluation of the model was done on a split of the training data.

## Metrics
The metrics used to evaluate the model performance were weighted averages of recall, precision and f1 score of the possible binary outputs.
Metrics were calculated on overall prediction as well as on data slices of categorical features.

## Ethical Considerations
This model was trained only for learning and demonstration purposes.
Its outputs shouldn't be considered for research or decision-making.

## Caveats and Recommendations
See above.
