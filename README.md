# CODSOFT Task 2: Credit Card Fraud Detection

A machine learning model that detects fraudulent credit card transactions, built for the **CodSoft Machine Learning Internship**.

**Author:** Saheli | [GitHub](https://github.com/itsmesaheli2005) | [LinkedIn](https://www.linkedin.com/in/saheli-sarkar-35097a3a5)

## Results

Evaluated on the full, untouched test set of **555,719 transactions**, of which only 2,145 (0.39%) are fraud. Fraud is the positive class.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 3.56% | 73.94% | 0.0680 | 0.1198 | 0.9096 |
| Decision Tree | 8.73% | 95.94% | 0.1600 | 0.2294 | 0.9852 |
| **Random Forest (best)** | **28.81%** | 92.96% | **0.4399** | **0.8291** | **0.9952** |

Models are ranked by **PR-AUC** (area under the precision-recall curve), the standard metric for highly imbalanced data. Accuracy is not used for ranking: a model that labelled every transaction "legitimate" would score 99.6%.

Best model, Random Forest, on the 555,719 test transactions:

- Caught **1,994 of 2,145** fraudulent transactions (92.96% recall).
- Raised **4,927 false alarms** among 553,574 legitimate transactions (0.89% of them).
- Missed 151 frauds.

**Trade-off:** at the default 0.5 threshold the model favours catching fraud over avoiding false alarms, so only 28.8% of its fraud alerts are real fraud. Part of this comes from rebalancing the training data. Raising the decision threshold would cut false alarms at the cost of some recall, and the right balance depends on the cost of a missed fraud versus a blocked legitimate purchase.

## What signals fraud?

The most influential features in the best model were **transaction amount (`amt`)**, **hour of the day**, and the categories **gas_transport**, **shopping_net** and **grocery_pos**. See `feature_importance.png`.

## How it works

1. **Load** `fraudTrain.csv` (1,296,675 transactions, 0.58% fraud) and `fraudTest.csv` (555,719 transactions).
2. **Engineer features:** hour, day of week and month from the timestamp, the customer's age from their date of birth, and the distance between the customer and the merchant (haversine formula).
3. **Drop** identifier and high-cardinality columns (card number, names, street, merchant, city, job, ...).
4. **Handle class imbalance:** keep all fraud rows and downsample legitimate rows to 60,000 for training (67,506 rows). The test set is never resampled, so the scores reflect real-world imbalance.
5. **Compare** Logistic Regression, Decision Tree and Random Forest, all with class weights.
6. **Select** the model with the best PR-AUC, plot the results and save it.

## Project structure

```
CODSOFT_2/
├── fraud_detector.py             # training + evaluation + scoring script
├── confusion_matrix.png          # confusion matrix on the test set
├── precision_recall_curves.png   # precision-recall curves for all three models
├── feature_importance.png        # top features driving the predictions
├── requirements.txt
└── README.md
```

## Dataset

Simulated credit card transactions dataset (linked in the CodSoft task sheet, also on Kaggle). Place both files in a `data/` folder:

```
data/
├── fraudTrain.csv
└── fraudTest.csv
```

The dataset files are about 480 MB together, so they are not included in this repo. The script also works with the single-file `creditcard.csv` dataset.

## How to run

```bash
pip install -r requirements.txt

# train, evaluate and save the model
python fraud_detector.py

# score a new file of transactions (same columns, without the is_fraud column)
python fraud_detector.py --score data/new_transactions.csv
```

Outputs are written to an `output/` folder: the trained model (`fraud_model.joblib`) and the three plots. The full run takes a few minutes, mostly for the Random Forest.

It also runs in Google Colab with no installs: copy the data into a `data/` folder and run `!python fraud_detector.py`.

## Possible improvements

- Tune the decision threshold to trade recall against false alarms.
- Try gradient boosting (XGBoost or LightGBM) and hyperparameter search.
- Add customer-level features such as spending history and transaction frequency.
- Use SMOTE or cost-sensitive learning instead of downsampling.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, joblib

## Demo

LinkedIn video: *add your link here after posting* #codsoft
