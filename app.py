import streamlit as st
import pickle
import numpy as np

# Load the trained pipeline and dataset
with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

with open('df.pkl', 'rb') as file:
    df = pickle.load(file)

# Define the layout of the Streamlit app
st.title("Property Price Prediction")

st.sidebar.header("Input Features")
property_type = st.sidebar.selectbox("Property Type", df['property_type'].unique())
sector = st.sidebar.selectbox("Sector", df['sector'].unique())
bedRoom = st.sidebar.slider("Number of Bedrooms", int(df['bedRoom'].min()), int(df['bedRoom'].max()), int(df['bedRoom'].mean()))
bathroom = st.sidebar.slider("Number of Bathrooms", int(df['bathroom'].min()), int(df['bathroom'].max()), int(df['bathroom'].mean()))
balcony = st.sidebar.selectbox("Balcony", df['balcony'].unique())
agePossession = st.sidebar.selectbox("Age Possession", df['agePossession'].unique())
built_up_area = st.sidebar.slider("Built-up Area (sq. ft.)", int(df['built_up_area'].min()), int(df['built_up_area'].max()), int(df['built_up_area'].mean()))
servant_room = st.sidebar.selectbox("Servant Room", df['servant room'].unique())
store_room = st.sidebar.selectbox("Store Room", df['store room'].unique())
furnishing_type = st.sidebar.selectbox("Furnishing Type", df['furnishing_type'].unique())
luxury_category = st.sidebar.selectbox("Luxury Category", df['luxury_category'].unique())
floor_category = st.sidebar.selectbox("Floor Category", df['floor_category'].unique())

# Create a dictionary with the user inputs
input_data = {
    'property_type': [property_type],
    'sector': [sector],
    'bedRoom': [bedRoom],
    'bathroom': [bathroom],
    'balcony': [balcony],
    'agePossession': [agePossession],
    'built_up_area': [built_up_area],
    'servant room': [servant_room],
    'store room': [store_room],
    'furnishing_type': [furnishing_type],
    'luxury_category': [luxury_category],
    'floor_category': [floor_category]
}

# Convert the input data into a DataFrame
input_df = pd.DataFrame(input_data)

# Predict the price using the pipeline
if st.button("Predict Price"):
    prediction = pipeline.predict(input_df)
    predicted_price = np.expm1(prediction)[0]  # Inverse of log1p transformation
    st.write(f"Predicted Property Price: ₹{predicted_price:,.2f}")

# Display some info about the dataset
st.write("### Dataset Overview")
st.write(df.head())
