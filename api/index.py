from flask import Flask, render_template, request
import joblib
import pandas as pd 
import numpy as np

from xgboost import XGBClassifier

app = Flask(__name__)

# Load model and scaler
model = XGBClassifier()
model.load_model("xgb_churn_model.json")
scaler = joblib.load("scaler.pkl")
best_threshold = joblib.load("best_threshold.pkl")
contract_encoder = joblib.load("contractlength_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
subscription_encoder = joblib.load("subscriptiontype_encoder.pkl")

# Feature column names (must match training order)
columns = [
    "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Subscription Type", "Contract Length",
    "Total Spend", "Days Since Last Interaction"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            age = float(request.form["Age"])
            gender = gender_encoder.transform([request.form["Gender"]])[0]
            tenure = float(request.form["Tenure"])
            usage_freq = float(request.form["UsageFrequency"])
            support_calls = int(float(request.form["SupportCalls"]))  # fixes '3.0' issue

            payment_delay = float(request.form["PaymentDelay"])
            subscription = subscription_encoder.transform([request.form["SubscriptionType"]])[0]
            contract = contract_encoder.transform([request.form["ContractLength"]])[0]
            total_spend = float(request.form["TotalSpend"])
            days_since = float(request.form["DaysSinceLastInteraction"])

            features = [
                age, gender, tenure, usage_freq, support_calls,
                payment_delay, subscription, contract, total_spend, days_since
            ]

            input_df = pd.DataFrame([features], columns=columns)
            scaled_input = scaler.transform(input_df)
            proba = model.predict_proba(scaled_input)[:, 1]

            # Ensure threshold is scalar
            threshold = float(best_threshold) if isinstance(best_threshold, (np.ndarray, list, tuple)) else best_threshold

            prediction = int((proba >= threshold).astype(int)[0])
            probability = round(proba[0] * 100, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, probability=probability)


if __name__ == "__main__":
    app.run(debug=True)


