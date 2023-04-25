# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Classification model developed for learning purpose.
The classification algorithm chosen was AdaBoostClassifier in its sklearn implementation, which consistently gave better performance than other alternatives.

## Intended Use
The model predicts salary based on biographic and ethnic data.
It is intended only to use for demonstration purposes.

## Training Data
Raw data is provided by the UCI Machine Learning Repository.
Details can be found here: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
Evaluation of the model was done on a split of the training data.
20% of the data was used for testing.

## Metrics
The metrics used to evaluate the model performance were weighted averages of recall, precision and f1 score of the possible binary outputs.
Metrics were calculated on overall prediction as well as on data slices of categorical features.
### Overall metrics
- Precision: 0.8529765975791673
- Recall: 0.8590511285122063
- Fbeta: 0.853385026016167

## Ethical Considerations
This model was trained only for learning and demonstration purposes.
The input data contains sensitive information, and it could lead do a misrepresantation of many categories as well as perpetuation of existing biases.
Its outputs shouldn't be considered for research or decision-making.

## Caveats and Recommendations
Since the training was done for demonstration only, it was done on limited data that doesn't accurately mirror the composition of the society.
Many categories have imbalanced data.
