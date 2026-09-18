
import gradio as gr
import joblib
import pandas as pd
import os

# Load model
model = joblib.load("multi_linear_regression_house_prediction_model.pkl")


def predict_price(area, bedrooms, floors):

    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Floors": [floors],
        
    })

    prediction = model.predict(input_data)[0]

    return f"Predicted Price: ₹{prediction:.2f} Lakhs"


demo = gr.Interface(
    fn=predict_price,

    inputs=[
        gr.Number(
            label="Enter Area (Sq Ft)",
            minimum=600,
            maximum=3000,
            value=600
        ),

        gr.Number(
            label="Enter No. of Bedrooms",
            minimum=1,
            maximum=4,
            value=3
        ),

        gr.Number(
            label="Enter No. of Floors",
            minimum=0,
            maximum=10,
            value=1
        )
    ],

    outputs=gr.Textbox(label="Prediction"),

    title="House Price Prediction",

    description="Predict house price based on area, bedrooms, and total floors."
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
