from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open(r"C:\Users\param\OneDrive\Desktop\ML\telecom_churn_app\model_91%_Accuracy.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    gender = int(request.form['gender'])
    age = float(request.form['age'])
    no_of_days_subscribed = float(request.form['no_of_days_subscribed'])
    multi_screen = int(request.form['multi_screen'])
    mail_subscribed = int(request.form['mail_subscribed'])
    weekly_mins_watched = float(request.form['weekly_mins_watched'])
    minimum_daily_mins = float(request.form['minimum_daily_mins'])
    maximum_daily_mins = float(request.form['maximum_daily_mins'])
    weekly_max_night_mins = float(request.form['weekly_max_night_mins'])
    videos_watched = float(request.form['videos_watched'])
    maximum_days_inactive = float(request.form['maximum_days_inactive'])
    customer_support_calls = float(request.form['customer_support_calls'])

    input_data = [[
        gender,
        age,
        no_of_days_subscribed,
        multi_screen,
        mail_subscribed,
        weekly_mins_watched,
        minimum_daily_mins,
        maximum_daily_mins,
        weekly_max_night_mins,
        videos_watched,
        maximum_days_inactive,
        customer_support_calls
    ]]

    prediction = model.predict(input_data)[0]

    result = "Customer will CHURN ❌" if prediction == 1 else "Customer will NOT churn ✅"

    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
