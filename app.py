import gradio as gr
import joblib
import numpy as np

#Load model and pipeline
model = joblib.load('lightgbm_model.pkl')
pipeline = joblib.load('pipeline.pkl')

#Category Mappings
sex_map = {"Male": 1, "Female": 0}
fbs_map = {"on or above 120 mg/dl": 1, "below 120 mg/dl": 0}
restecg_map = {"Normal": 0, "T wave inversion and/or ST elevation/depression > 0.05mV": 1, "Left Ventricular Hypertrophy (by Estes' criteria)": 2}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
cp_map = {"Asymptomatic": 0, "Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3}
thal_map = { "Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

# Prediction function
def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    try:
        if not (0 <= int(ca) <= 3):
            return " Number of major vessels must be between 0 and 3."
        if int(age) <= 0 or float(trestbps) <= 0 or float(chol) <= 0 or float(thalach) <= 0:
            return "Age, BP, Cholesterol, and Heart Rate must be positive numbers."

        mapped_inputs = [
            int(age),
            sex_map[sex],
            cp_map[cp],
            float(trestbps),
            float(chol),
            fbs_map[fbs],
            restecg_map[restecg],
            float(thalach),
            exang_map[exang],
            float(oldpeak),
            slope_map[slope],
            int(ca),
            thal_map[thal]
        ]

        X = np.array([mapped_inputs])
        X_transformed = pipeline.transform(X)
        pred = model.predict(X_transformed)[0]

        return "Hypertension Detected. Please consult a doctor." if pred == 1 else "No Hypertension Detected."

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='color:#6a0dad; text-align:center;'>Hypertension Risk Predictor</h1>")
    gr.Markdown("<p style='text-align:center; color:#a34acb;'>Enter the details below to assess your hypertension risk.</p>")

    with gr.Row():
        age = gr.Number(label="Age (years)", minimum=1)
        sex = gr.Dropdown(list(sex_map.keys()), label="Sex")
        cp = gr.Dropdown(list(cp_map.keys()), label="Chest Pain Type")

    with gr.Row():
        trestbps = gr.Number(label="Resting BP (mmHg)")
        chol = gr.Number(label="Serum Cholesterol (mg/dl)")
        fbs = gr.Dropdown(list(fbs_map.keys()), label="Fasting Blood Sugar")

    with gr.Row():
        restecg = gr.Dropdown(list(restecg_map.keys()), label="Resting ECG Result")
        thalach = gr.Number(label="Max Heart Rate")
        exang = gr.Dropdown(list(exang_map.keys()), label="Exercise-Induced Angina")

    with gr.Row():
        oldpeak = gr.Number(label="ST Depression")
        slope = gr.Dropdown(list(slope_map.keys()), label="ST Slope")
        ca = gr.Number(label="No. of Major Vessels (0–3)", precision=0)
        thal = gr.Dropdown(list(thal_map.keys()), label="Thalassemia Type")

    submit_btn = gr.Button("💡 Predict")
    output = gr.Textbox(label="Prediction Result", lines=2)
    submit_btn.click(
        fn=predict,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        outputs=output
    )

    gr.Markdown(
        "<br><hr><p style='font-size: 13px; text-align:center; color:#999;'>"
        "This tool is for informational purposes only and does not replace professional medical advice."
        "</p>"
    )

demo.launch()