# 🧠 Customer Churn Prediction App

A web-based application to predict whether a customer will churn or not using a machine learning model trained in `churn.ipynb` and deployed using Flask (`app.py`) with a user-friendly HTML/CSS frontend.




## 🎯 Objective

To predict whether a customer will churn based on:
- Demographic info (age, gender)
- Subscription details
- Usage patterns
- Support history
- Payment behavior

---

## 🧪 Model Training (churn.ipynb)

1. Data preprocessing
2. SMOTE for class balancing
3. Feature scaling
4. Model training using `XGBClassifier`
5. Threshold tuning for best precision/recall
6. Model evaluation
7. Export model and scaler using `joblib`

---

## 🚀 Flask App (app.py)

- Loads the trained model and scaler
- Accepts user input via HTML form
- Encodes and scales the input
- Predicts the probability of churn
- Displays prediction result on the same page

---

## 💡 Features

- ✔️ Clean web interface with HTML/CSS
- ✔️ Real-time churn prediction
- ✔️ Probability-based output
- ✔️ Editable inputs for all features
- ✔️ Threshold tuning for optimization (optional)

---

## 🖥️ How to Run the App Locally

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/yourusername/churn-prediction-app.git
cd churn-prediction-app
```

### 🐍 2. Create and Activate Environment (optional)

```bash
conda create -n churn_env python=3.9
conda activate churn_env
```

### 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🧠 4. Run the Flask App

```bash
python app.py
```

Then open your browser and visit:  
📍 `http://127.0.0.1:5000/`

---

## 🧾 Example Input Features

- `Age`: 22 - 70
- `Gender`: Male / Female
- `Tenure`: Months since joining
- `Usage Frequency`: How often the service is used
- `Support Calls`: Number of support tickets
- `Payment Delay`: Days payment was delayed
- `Subscription Type`: Basic, Standard, Premium
- `Contract Length`: Monthly, Quarterly, Annual
- `Total Spend`: Total amount paid
- `Last Interaction`: Days since last interaction

---

## 📊 Sample Output

```bash
🔍 Churn Probability: 0.84
🛑 Customer is likely to churn
```

---

## 📌 Dependencies

- Python 3.8+
- Flask
- scikit-learn
- xgboost
- imbalanced-learn
- joblib
- pandas, numpy
- HTML5 & CSS3

---

## 📜 License

MIT License

---

## 🙋‍♂️ Author

**Muhammad Shoaib Sattar**  
Email: mshoaib3393@gmail.com  
GitHub: [shoaib1-coder](https://github.com/shoaib1-coder)
