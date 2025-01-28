import streamlit as st
import pickle
import numpy as np

# Load the trained models
encoder_model = pickle.load(open(r"C:\Users\Ranjan kumar pradhan\.vscode\project_vs\diamonds_price\label_encoder.pkl", "rb"))
prediction_model = pickle.load(open(r"C:\Users\Ranjan kumar pradhan\.vscode\project_vs\diamonds_price\model.pkl", "rb"))

# Streamlit app
def main():
    st.title("Diamond Price Predictor")
    st.write("Enter the details of the diamond to predict its price:")

    # Input fields
    carat = st.number_input("Carat", min_value=0.0, step=0.01, format="%.2f")

    cut = st.selectbox("Cut", options=['Ideal', 'Premium', 'Good', 'Very Good', 'Fair'])
    color = st.selectbox("Color", options=['E', 'I', 'J', 'H', 'F', 'G', 'D'])
    clarity = st.selectbox("Clarity", options=['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF'])

    depth = st.number_input("Depth %", min_value=0.0, step=0.1, format="%.1f")
    table = st.number_input("Table %", min_value=0.0, step=0.1, format="%.1f")

    x = st.number_input("X (mm)", min_value=0.0, step=0.01, format="%.2f")
    y = st.number_input("Y (mm)", min_value=0.0, step=0.01, format="%.2f")
    z = st.number_input("Z (mm)", min_value=0.0, step=0.01, format="%.2f")

    # Predict button
    if st.button("Predict Price"):
        try:
            # Prepare the input array for encoding
            input_data = np.array([[carat, cut, color, clarity, depth, table, x, y, z]], dtype=object)

            # Convert categorical columns to numerical using encoder model
            input_data[:, 1:4] = encoder_model.transform(input_data[:, 1:4])

            # Predict price using the prediction model
            predicted_price = prediction_model.predict(input_data)[0]

            st.success(f"Predicted Price: ${predicted_price:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()

