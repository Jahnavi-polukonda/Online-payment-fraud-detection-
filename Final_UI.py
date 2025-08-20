import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model_path = r"C:\Users\jahan\Downloads\optimized_model.keras"
model = load_model(model_path)

# Page configuration
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Home", "About Us", "Prediction", "History"])

# Initialize or load history in session state
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=[
        'Step', 'Amount', 'Old Balance Origin', 'New Balance Origin', 
        'Old Balance Destination', 'New Balance Destination', 'Type', 'Prediction'
    ])

# Custom CSS for styling
st.markdown("""
    <style>
        .section-title {
            background-color: #003366;
            color: #FFFFFF;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .section-content {
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Home Section
if section == "Home":
    st.markdown("<h1 class='section-title'>Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Welcome to the Fraud Detection System</h2>", unsafe_allow_html=True)

    image_path = r'C:\Users\jahan\image.jpg'
    image = Image.open(image_path)
    resized_image = image.resize((600, 300))
    st.image(resized_image, use_container_width=False)

    st.markdown("""
        <div class='section-content'>
            <p>Our system helps you detect fraudulent transactions in real-time, ensuring your online transactions are safe and secure.</p>
            <ul>
                <li><b>Real-time Fraud Detection</b>: Instantly analyze transactions to identify potential fraudulent activity.</li>
                <li><b>Advanced Machine Learning Models</b>: Leverage sophisticated algorithms trained on extensive historical data.</li>
                <li><b>User-Friendly Interface</b>: Simple and intuitive design that requires minimal training for users.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# About Us Section
elif section == "About Us":
    st.markdown("<h1 class='section-title'>About Us</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class='section-content'>
            <p>We are committed to providing a powerful and efficient system for detecting fraud in financial transactions.</p>
            <ul>
                <li><b>Streamlit</b> for building the user interface.</li>
                <li><b>Keras</b> for developing and integrating the machine learning model.</li>
                <li><b>Pandas</b> for data handling and processing.</li>
            </ul>
             <p>Overview of the Fraud Detection System App Design
1. Application Framework
Streamlit: We used Streamlit as the core framework to build the web application due to its simplicity, interactive capabilities, and rapid prototyping features. It enables seamless integration with Python scripts and offers intuitive, built-in components for UI development.

2. Model Integration
Keras Model: The core predictive engine of the application is powered by a pre-trained Keras model loaded using the load_model() function from TensorFlow/Keras. This allows the app to make real-time predictions based on user input.

3. Custom Layout and Styling
Page Layout: To create a modern, multi-tab-like layout, we utilized st.columns() for horizontal alignment, simulating a navigation bar without requiring external navigation components. This approach improves user interaction and keeps content easily accessible.
st.container(): Each section's content (e.g., Home, About Us, Prediction) is wrapped within st.container() blocks, allowing us to organize and display content conditionally based on user interaction with buttons.
CSS Styling: We included custom HTML and CSS directly in the st.markdown() function to enhance the appearance of the app, adding a professional touch through color themes, borders, and background changes.

4. Design Elements
Headers and Sections: Headers are styled with st.markdown() to incorporate custom HTML and CSS, giving them a unique color scheme and rounded edges.
Icons and Buttons: We added icons to the buttons using Unicode characters for a more user-friendly and visually appealing navigation experience.
Interactive Components: Input fields (st.number_input() and st.selectbox()) are used for users to provide transaction details, which are then processed by the model for predictions.

5. User Experience Enhancements
Alerts and Notifications: Based on the model's prediction, the app uses st.success() and st.error() to provide immediate feedback, with customized messages and icons to convey the results effectively.
Minimalistic Design: The interface is clean and intuitive, using soft colors and padding to create a balanced layout that’s easy on the eyes.

Why This Approach?

Ease of Use: Using Streamlit and Python allows the app to be simple and accessible for users who may not be tech-savvy. The intuitive UI ensures that users can input data and get results quickly.

Customization: The use of custom HTML/CSS in st.markdown() allows for unique design elements without needing a separate front-end development phase.

Interactivity: The app's interactive buttons and input fields enhance user engagement, making it easy to switch between sections and enter data seamlessly.

Technologies Used:

Streamlit: For building the web interface.

TensorFlow/Keras: For loading the trained model and making predictions.

Pandas: For managing input data and data structures.

HTML/CSS: Embedded within st.markdown() for custom design.</p>
        </div>
    """, unsafe_allow_html=True)

# Prediction Section
elif section == "Prediction":
    st.markdown("<h1 class='section-title'>Transaction Fraud Prediction</h1>", unsafe_allow_html=True)
    prediction_type = st.radio("Select Prediction Type", ["Single Prediction", "Batch Prediction"])

    if prediction_type == "Single Prediction":
        st.subheader("Single Prediction")
        with st.form(key='single_prediction_form'):
            step = st.number_input("Step", min_value=0)
            amount = st.number_input("Amount", min_value=0.0)
            oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0)
            newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0)
            oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0)
            newbalanceDest = st.number_input("New Balance Destination", min_value=0.0)
            type_transaction = st.selectbox("Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
            submit_button = st.form_submit_button("Predict")

            if submit_button:
                # Encode the transaction type
                type_dict = {
                    "CASH_IN": [1, 0, 0, 0, 0],
                    "CASH_OUT": [0, 1, 0, 0, 0],
                    "DEBIT": [0, 0, 1, 0, 0],
                    "PAYMENT": [0, 0, 0, 1, 0],
                    "TRANSFER": [0, 0, 0, 0, 1]
                }
                type_encoded = type_dict.get(type_transaction, [0, 0, 0, 0, 0])
                input_data = [[step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest] + type_encoded]

                # Predict
                prediction = model.predict(np.array(input_data))
                result = "🚨 ALERT: Fraudulent Transaction Detected" if prediction[0][0] > 0.5 else "✅ Transaction is NOT Fraudulent."
                st.write(f"### Prediction: {result}")

                # Save to history
                new_entry = pd.DataFrame([[step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type_transaction, result]],
                                         columns=['Step', 'Amount', 'Old Balance Origin', 'New Balance Origin', 
                                                  'Old Balance Destination', 'New Balance Destination', 'Type', 'Prediction'])
                st.session_state['history'] = pd.concat([st.session_state['history'], new_entry], ignore_index=True)

    elif prediction_type == "Batch Prediction":
        st.subheader("Batch Prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            required_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type']
            if all(col in data.columns for col in required_columns):
                type_dict = {
                    "CASH_IN": [1, 0, 0, 0, 0],
                    "CASH_OUT": [0, 1, 0, 0, 0],
                    "DEBIT": [0, 0, 1, 0, 0],
                    "PAYMENT": [0, 0, 0, 1, 0],
                    "TRANSFER": [0, 0, 0, 0, 1]
                }

                processed_data = []
                for index, row in data.iterrows():
                    type_encoded = type_dict.get(row['type'], [0, 0, 0, 0, 0])
                    input_row = [row['step'], row['amount'], row['oldbalanceOrg'], row['newbalanceOrig'], row['oldbalanceDest'], row['newbalanceDest']] + type_encoded
                    processed_data.append(input_row)

                processed_data = pd.DataFrame(processed_data)
                predictions = model.predict(processed_data)
                data['Prediction'] = (predictions > 0.5).astype(int)

                st.write("### Prediction Results")
                st.write(data)
                st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv", "text/csv")
            else:
                st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")

# History Section
elif section == "History":
    st.markdown("<h1 class='section-title'>Prediction History</h1>", unsafe_allow_html=True)
    if not st.session_state['history'].empty:
        st.dataframe(st.session_state['history'])
        st.download_button("Download History", st.session_state['history'].to_csv(index=False), "history.csv", "text/csv")
    else:
        st.write("No prediction history available.")
