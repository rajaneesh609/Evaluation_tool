import gradio as gr
import numpy as np
import joblib
from sklearn.linear_model import Perceptron

import pandas as pd

file_path = "Student-Employability-Datasets (1).xlsx"
df = pd.read_excel(file_path, sheet_name='Data')

X = df.iloc[:, 1:8].values 
Y = (df['CLASS'] == 'Employable').astype(int).values


model = Perceptron()
model.fit(X, Y)

# Save model
joblib.dump(model, "employability_model.pkl")


def evaluate_employability(name, *inputs):
    model = joblib.load("employability_model.pkl")
    inputs = np.array(inputs).reshape(1, -1)
    prediction = model.predict(inputs)[0]
    if prediction == 1:
        return f"Congrats {name}! 🎉 You are employable!"
    else:
        return f"Try to upgrade yourself, {name}! 💪 Keep improving!"

# UI with sliders
demo = gr.Interface(
    fn=evaluate_employability,
    inputs=[
        gr.Textbox(label="Enter your name"),
        gr.Slider(1, 5, label="GENERAL APPEARANCE"),
        gr.Slider(1, 5, label="MANNER OF SPEAKING"),
        gr.Slider(1, 5, label="PHYSICAL CONDITION"),
        gr.Slider(1, 5, label="MENTAL ALERTNESS"),
        gr.Slider(1, 5, label="SELF-CONFIDENCE"),
        gr.Slider(1, 5, label="ABILITY TO PRESENT IDEAS"),
        gr.Slider(1, 5, label="COMMUNICATION SKILLS"),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Employment Capability Assessment",
    description="Rate yourself and see your employability potential!"
)

demo.launch()
