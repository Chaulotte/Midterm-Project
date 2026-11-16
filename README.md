# Palmer Penguins Sex Classification

## 1. Problem Framing

### What We Are Predicting
This project predicts the **biological sex** (`male` vs. `female`) of Antarctic penguins from the Palmer Archipelago using non-invasive physical measurements (`culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`) and habitat attributes (`species`, `island`).

- **Target Variable:** `sex` (`1` = male, `0` = female)
- **Task Type:** Balanced Binary Classification (50.5% male, 49.5% female)

### Who Benefits & How the Model Will Be Used
- **Target Users:** Field wildlife biologists, Antarctic ecological researchers, and marine conservation teams.
- **Practical Usage:** During routine field banding and health checks, researchers measure penguin bill dimensions, flipper length, and body mass. Instead of waiting weeks for lab results, field applications or automated data pipelines can query this containerized REST API (`POST /predict`) to obtain instant sex probability estimates on site.

### Why This Problem Matters
Determining sex in penguins is difficult visually because males and females share nearly identical plumage (monomorphic coloration). Traditional genetic sexing requires drawing blood or plucking feathers, which is invasive, expensive, and logistically difficult in extreme Antarctic field conditions. Accurate, real-time sex classification enables researchers to:
- Monitor colony sex ratios and breeding pair dynamics without harming birds.
- Track sex-specific foraging behavior, survival rates, and climate-change impacts across species.

### Evaluation Metric
- **Primary Metric:** **ROC AUC** (Receiver Operating Characteristic Area Under the Curve).
- **Justification:** Because the cleaned dataset is well-balanced (`50.5%` male vs. `49.5%` female), ROC AUC provides a threshold-independent measure of how reliably the model ranks male specimens above female specimens across all classification cutoffs.

---

## 2. Dataset Description
The dataset (`data/penguins_size.csv`) comes from the Palmer Archipelago (Antarctica) penguin dataset (344 rows, 7 columns):

| Feature | Type | Description |
| :--- | :--- | :--- |
| `species` | Categorical | Penguin species (`Adelie`, `Chinstrap`, `Gentoo`) |
| `island` | Categorical | Island in Palmer Archipelago (`Biscoe`, `Dream`, `Torgersen`) |
| `culmen_length_mm` | Numerical | Culmen (bill) length in millimeters |
| `culmen_depth_mm` | Numerical | Culmen (bill) depth in millimeters |
| `flipper_length_mm` | Numerical | Flipper length in millimeters |
| `body_mass_g` | Numerical | Body mass in grams |
| `sex` | Target | Biological sex (`MALE`, `FEMALE`) |

After removing 10 rows with missing `sex` values and 1 invalid `"."` entry, **333 clean records** remain.

---

## 3. Exploratory Data Analysis (EDA) Summary
- **Sexual Dimorphism:** Across all three species, males consistently exhibit higher body mass, deeper culmens, and longer culmens than females.
- **Mutual Information:** `culmen_length_mm` (`0.4839`) and `body_mass_g` (`0.4045`) carry the strongest individual predictive signal for `sex`, followed by `culmen_depth_mm` (`0.3669`) and `flipper_length_mm` (`0.2658`).
- **Feature Correlation:** `flipper_length_mm` and `body_mass_g` show strong positive correlation (`0.87`), reflecting overall body size scaling across species.

---

## 4. Modeling Approach & Metrics

The cleaned dataset was split into **60% train / 20% validation / 20% test** (`random_state=1`), with 5-fold cross-validation performed on the combined 80% training set (`266` samples) prior to final evaluation on the 20% holdout test set (`67` samples).

| Model | 5-Fold CV ROC AUC | CV Std | Test ROC AUC |
| :--- | :---: | :---: | :---: |
| **Random Forest (`n_estimators=100, max_depth=5, min_samples_leaf=2`)** | **0.9745** | **± 0.0136** | **0.9642** |
| Gradient Boosting (`lr=0.1, max_depth=3, n_estimators=100`) | 0.9749 | ± 0.0170 | 0.9391 |
| Logistic Regression (`C=1.0`) | 0.9702 | ± 0.0175 | 0.9686 |
| Decision Tree (`max_depth=4, min_samples_leaf=10`) | 0.9131 | ± 0.0362 | 0.9516 |

### Model Selection Rationale
**Random Forest (`n_estimators=100, max_depth=5, min_samples_leaf=2`)** was selected as the final production model because it achieved top cross-validation performance (**0.9745**) with the **lowest fold-to-fold variance (`± 0.0136`)** and strong holdout test generalization (**0.9642**), avoiding the overfitting observed in Gradient Boosting on a 333-row dataset.

---

## 5. System Architecture

```text
data/penguins_size.csv ──▶ model_train.py ──▶ model.bin ──▶ model_predict.py (Flask + Gunicorn) ──▶ POST /predict
```

---

## 6. How to Run Locally

### Option A: Using `pipenv`
```bash
pip install pipenv
pipenv install --dev
pipenv shell
```

### Option B: Using `pip`
```bash
pip install -r requirements.txt
```

### Train the Model & Start the Web Service
1. Train and export `model.bin`:
   ```bash
   python model_train.py
   ```
2. Start the Flask service on port `9696`:
   ```bash
   python model_predict.py
   ```
3. In a second terminal, run the test suite:
   ```bash
   python test.py
   ```

---

## 7. Run with Docker

### Build the Docker Image
```bash
docker build -t penguin-sex-predictor .
```

### Run the Container
```bash
docker run -it --rm -p 9696:9696 penguin-sex-predictor
```

### Verify Endpoints
```bash
python test.py
```

---

## 8. API Usage Example

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
     -d '{"species": "Adelie", "island": "Torgersen", "culmen_length_mm": 39.1, "culmen_depth_mm": 18.7, "flipper_length_mm": 181.0, "body_mass_g": 3750.0}' \
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

---

## 9. Known Limitations & Next Steps
- **Dataset Size:** With 333 clean records, predictions are sensitive to rare measurements within smaller subgroups (such as Chinstrap penguins). Expanding the dataset with multi-season field records would further reduce variance.
- **Next Steps:** Add Pydantic request schema validation for malformed inputs, incorporate confidence intervals for borderline predictions, and deploy the Docker container to a cloud platform (Render, Railway, or AWS).
