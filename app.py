from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# --------- LOAD TRAINED MODEL & COLUMNS ---------
model = joblib.load("diabetes_nb_model.pkl")
model_columns = joblib.load("model_columns.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            pregnancies = float(request.form["pregnancies"])
            glucose = float(request.form["glucose"])
            blood_pressure = float(request.form["blood_pressure"])
            skin_thickness = float(request.form["skin_thickness"])
            insulin = float(request.form["insulin"])
            bmi = float(request.form["bmi"])
            dpf = float(request.form["dpf"])
            age = float(request.form["age"])

            # Build input in same column order as training
            input_dict = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age
            }

            input_data = pd.DataFrame([input_dict])[model_columns]

            pred_class = model.predict(input_data)[0]
            pred_proba = model.predict_proba(input_data)[0][pred_class]

            if pred_class == 1:
                prediction = "High Risk (Likely Diabetic)"
            else:
                prediction = "Low Risk (Not Diabetic)"

            probability = round(pred_proba * 100, 2)

        except Exception as e:
            prediction = f"Error: {e}"
            probability = None

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability)


if __name__ == "__main__":
    app.run(debug=True)
