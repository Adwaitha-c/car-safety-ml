import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the Decision Tree model and scaler
with open('trained_model_dt.pkl', 'rb') as model_file:
    dt = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the encoders
encoders = {}
for column in ['buying_price', 'maint_cost', 'doors', 'persons', 'lug_boot', 'safety']:
    with open(f'le_{column}.pkl', 'rb') as file:
        encoders[column] = pickle.load(file)

st.title("Car Evaluation Model")

# Sidebar for user input
st.sidebar.header("Input Features")
def user_input_features():
    buying_price = st.sidebar.selectbox("Buying Price", encoders['buying_price'].classes_)
    maint_cost = st.sidebar.selectbox("Maintenance Cost", encoders['maint_cost'].classes_)
    doors = st.sidebar.selectbox("Number of Doors", encoders['doors'].classes_)
    persons = st.sidebar.selectbox("Number of Persons", encoders['persons'].classes_)
    lug_boot = st.sidebar.selectbox("Luggage Boot Size", encoders['lug_boot'].classes_)
    safety = st.sidebar.selectbox("Safety", encoders['safety'].classes_)

    data = {
        'buying_price': encoders['buying_price'].transform([buying_price])[0],
        'maint_cost': encoders['maint_cost'].transform([maint_cost])[0],
        'doors': encoders['doors'].transform([doors])[0],
        'persons': encoders['persons'].transform([persons])[0],
        'lug_boot': encoders['lug_boot'].transform([lug_boot])[0],
        'safety': encoders['safety'].transform([safety])[0]
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Scale the input features
input_scaled = scaler.transform(input_df)

# Model prediction
model = dt
prediction = model.predict(input_df)

# Define the prediction mapping
prediction_mapping = {
    0: "Unacceptable",
    1: "Acceptable"
}

# Display prediction
st.subheader("Decision Tree Prediction")
st.write(f"Prediction: {prediction_mapping.get(prediction[0], 'Unknown')}")

# Run the app with `streamlit run app.py`
