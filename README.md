# Palmer Penguins Sex Classification

## 1. Problem Framing

### Problem Description
Field researchers studying Antarctic penguins frequently record external body measurements without performing invasive biological sampling. This project solves a **binary classification** problem: predicting the biological sex (`male` vs. `female`) of Palmer Archipelago penguins from non-invasive morphological measurements (`culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`) and habitat location (`species`, `island`).

### Who Benefits & How It Is Used
- **Primary Users:** Wildlife biologists, ecological researchers, and Antarctic conservation teams.
- **Practical Application:** Researchers enter standard field measurements captured during routine tagging and weighing into a lightweight REST API (`POST /predict`) to obtain immediate, non-invasive sex predictions and probability scores.

### Why This Problem Matters
Determining sex ratios in wild penguin colonies is essential for modeling population growth, reproductive success, and foraging behavior under changing climate conditions. Traditional DNA or blood sexing requires specialized lab equipment, permits, and stressful handling of birds. An accurate morphological classifier provides instant, zero-cost sex estimation directly in the field.

### Evaluation Metric
- **Metric:** **ROC AUC** (Receiver Operating Characteristic Area Under the Curve).
- **Justification:** After cleaning, the dataset is well-balanced (**50.5% male**, **49.5% female**). ROC AUC evaluates ranking and probability calibration across all classification thresholds, making it ideal for balanced binary classification where field researchers may adjust probability cutoffs for high-confidence filtering.

---

## 2. Dataset Description
The dataset (`data/penguins_size.csv`) originates from the Palmer Archipelago (Antarctica) Penguin dataset collected by Dr. Kristen Gorman and the Palmer Station LTER (344 rows, 7 columns):

| Feature | Type | Description |
| :--- | :--- | :--- |
| `species` | Categorical | Penguin species (`Adelie`, `Chinstrap`, `Gentoo`) |
| `island` | Categorical | Island in Palmer Archipelago (`Biscoe`, `Dream`, `Torgersen`) |
| `culmen_length_mm` | Numerical | Culmen (bill) length in millimeters |
| `culmen_depth_mm` | Numerical | Culmen (bill) depth in millimeters |
| `flipper_length_mm` | Numerical | Flipper length in millimeters |
| `body_mass_g` | Numerical | Body mass in grams |
| `sex` (Target) | Binary | Biological sex (`MALE` -> `1`, `FEMALE` -> `0`) |

**Data Cleaning:** Removed 10 rows with missing `sex` values and 1 invalid `"."` entry, leaving **333 complete records** split 60% train (199) / 20% validation (67) / 20% test (67).

---

## 3. Exploratory Data Analysis & Modeling Summary

### Key EDA Insights
- **Sexual Dimorphism:** Across all three species, males consistently exhibit greater body mass, deeper culmens, and longer bills than females.
- **Feature Importance:** Mutual information analysis shows `culmen_length_mm` (`0.484`) and `body_mass_g` (`0.405`) carry the strongest individual predictive signal with respect to `sex`.

### Visual Insights
![Feature Distributions by Species and Sex](assets/feature_distributions.png)

![Correlation Heatmap](assets/correlation_heatmap.png)

### Model Comparison & Selection Rationale
Four model families were trained, tuned, and evaluated using **5-fold cross-validation** on the training set and verified on the holdout test set:

| Model | 5-Fold CV ROC AUC | CV Std | Test ROC AUC |
| :--- | :---: | :---: | :---: |
| **Random Forest (`n_estimators=100, max_depth=5, min_samples_leaf=2`)** | **0.9745** | **± 0.0136** | **0.9642** |
| Gradient Boosting (`lr=0.1, max_depth=3, n_estimators=100`) | 0.9749 | ± 0.0170 | 0.9391 |
| Logistic Regression (`C=1.0`) | 0.9702 | ± 0.0175 | 0.9686 |
| Decision Tree (`max_depth=4, min_samples_leaf=10`) | 0.9131 | ± 0.0362 | 0.9516 |

**Why Random Forest Was Selected:**
- Achieved top-tier 5-fold cross-validation performance (**0.9745**) with the **lowest fold-to-fold standard deviation (`± 0.0136`)**.
- Generalized reliably to the unseen test set (**0.9642**), avoiding the overfitting observed in Gradient Boosting (`0.9749` CV -> `0.9391` test) on this 333-row dataset.

---

## 4. Project Architecture & Structure

```text
data/penguins_size.csv ──▶ model_train.py ──▶ model.bin ──▶ model_predict.py (Flask + Gunicorn) ──▶ POST /predict
```

```text
Midterm-Project/
├── assets/                 # EDA plots and API demo screenshots
├── data/
│   └── penguins_size.csv   # Palmer Penguins dataset
├── Dockerfile              # Container definition (Python 3.11 + Gunicorn)
├── LICENSE                 # MIT License
├── model.bin               # Serialized DictVectorizer + Random Forest model
├── model_predict.py        # Flask web service (/health and /predict)
├── model_train.py          # Reproducible training & model export script
├── notebook.ipynb          # Complete EDA, tuning, CV, and model selection
├── Pipfile                 # Pipenv environment specification
├── Pipfile.lock            # Locked dependency versions
├── README.md               # Project documentation
├── requirements.txt        # Pip requirements for Docker and local setup
├── test.ipynb              # Interactive notebook API tests
└── test.py                 # Automated CLI script to test endpoints
```

---

## 5. How to Run Locally

### 1. Set Up Environment
Using `pipenv`:
```bash
pip install pipenv
pipenv install --dev
pipenv shell
```
Or using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python model_train.py
```

### 3. Start the Web Service
```bash
python model_predict.py
```
The service starts on `http://localhost:9696`.

---

## 6. Run with Docker

### Build the Image
```bash
docker build -t penguin-sex-predictor .
```

### Run the Container
```bash
docker run -it --rm -p 9696:9696 penguin-sex-predictor
```

---

## 7. API Usage Example

### Health Check (`GET /health`)
```bash
curl http://localhost:9696/health
```
Response:
```json
{"status": "ok"}
```

### Predict Sex (`POST /predict`)
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{
           "species": "Adelie",
           "island": "Torgersen",
           "culmen_length_mm": 39.1,
           "culmen_depth_mm": 18.7,
           "flipper_length_mm": 181.0,
           "body_mass_g": 3750.0
         }' \
     http://localhost:9696/predict
```
Response:
```json
{
  "is_male": true,
  "male_probability": 0.5687,
  "predicted_sex": "male"
}
```

You can also run the automated test script in a separate terminal:
```bash
python test.py
```

![API Demo](assets/api_demo.png)

---

## 8. Known Limitations & Next Steps
- **Small Sample Size:** With 333 valid records, individual outliers within smaller subgroups (such as `Chinstrap` on `Dream` island) can influence split thresholds.
- **Next Steps:**
  - Add Pydantic request schema validation for malformed inputs or out-of-range measurements.
  - Deploy the Docker container to a cloud hosting service (Render, Railway, or AWS Elastic Beanstalk) with CI/CD checks.
