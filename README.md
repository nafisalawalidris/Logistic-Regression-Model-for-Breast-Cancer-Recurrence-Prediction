# Logistic Regression Model to Predict Breast Cancer Recurrence

Author: [**Nafisa Lawal Idris**]

## Scenario
Breast cancer is a leading cause of death, affecting millions of women worldwide. Early detection of recurrence can significantly improve a woman's chance of survival. This project aims to create a logistic regression model to classify patients at greater risk for breast cancer recurrence based on various attributes.

## Data Files
- `~/Projects/LogisticRegression.ipynb`
- `~/Projects/breast_cancer_data/breast-cancer.csv`

## Dataset Attributes
- **recurrence**: Whether the patient had a recurrence event. (0 - No recurrence, 1 - Recurrence)
- **age_decade**: Age of the patient at the time of diagnosis, divided into bins for each decade.
- **meno_pre**: Whether the patient has not yet reached menopause.
- **meno_lt_40**: Whether the patient was less than 40 when reaching menopause.
- **meno_ge_40**: Whether the patient was at least 40 when reaching menopause.
- **tumor_size**: The largest diameter (in millimeters) of the excised tumor.
- **inv_nodes**: The number of axillary lymph nodes containing metastatic breast cancer in a histological examination.
- **node_caps**: Whether the cancer has spread outside the lymph node capsule.
- **deg_malig**: The histological grade of the tumor. (1 - Still largely normal, 2 - Somewhat abnormal, 3 - Largely abnormal)
- **breast_left**: Whether the patient had cancer in the left breast.
- **breast_right**: Whether the patient had cancer in the right breast.
- **irradiat**: Whether the cancer has been irradiated. (0 - Not irradiated, 1 - Irradiated)

## Results
After preprocessing and training a logistic regression model on the dataset, the model achieved an accuracy of 68.97% on the validation set. The model can now be used to predict breast cancer recurrence in new patients based on their attributes.

## Usage
The provided Jupyter notebook (LogisticRegression.ipynb) contains the code used for data preprocessing, model training, and evaluation. The dataset (breast-cancer.csv) used for this project is located in the breast_cancer_data directory. You can follow the notebook's instructions to reproduce the results or modify the code for further experimentation.

## Contribution
Feel free to contribute to this project by improving the model, exploring different machine learning algorithms, or incorporating additional relevant features. Your contributions are welcome and greatly appreciated!

## License
This project is licensed under [MIT].
