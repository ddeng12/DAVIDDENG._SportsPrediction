import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import NotFittedError
import os

st.title("FIFA Model Deployment")

model_file = 'best_model.pkl'
scaler_file = 'scaler (1).pkl'

st.write("Files in the current directory:")
st.write(os.listdir('.'))

try:
    with open(model_file, 'rb') as f:
        best_model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file '{model_file}' not found. Please ensure the file is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error(f"Scaler file '{scaler_file}' not found. Please ensure the file is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

try:
    _ = scaler.mean_
except AttributeError:
    st.error("The scaler is not fitted yet. Please fit the scaler before using it.")
    st.stop()

def preprocess_data(data, train_columns):
    for col in train_columns:
        if col not in data.columns:
            data[col] = 0
    data = data[train_columns]
    data = data.fillna(0)  
    return data

def test_model_in_batches(model, X_new, y_new, train_columns, scaler=None, batch_size=1000):
    num_batches = len(X_new) // batch_size + 1
    all_predictions = []

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        X_batch = X_new.iloc[start:end]  
        y_batch = y_new[start:end]

        X_batch = preprocess_data(X_batch, train_columns)

        if scaler:
            try:
                X_batch = pd.DataFrame(scaler.transform(X_batch), columns=X_batch.columns)
            except NotFittedError:
                st.error("The scaler is not fitted yet. Please fit the scaler before using it.")
                st.stop()

        y_pred_batch = model.predict(X_batch)
        all_predictions.extend(y_pred_batch)

    mse = mean_squared_error(y_new, all_predictions)
    r2 = r2_score(y_new, all_predictions)

    return {
        'mean_squared_error': mse,
        'r2_score': r2
    }

uploaded_file = st.file_uploader("Choose a CSV file for testing", type="csv")
if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)
        
        train_columns = ['value_eur', 'wage_eur', 'potential', 'age', 'defending', 'defending_standing_tackle',
                         'mentality_interceptions', 'defending_sliding_tackle', 'international_reputation',
                         'movement_reactions', 'defending_marking_awareness', 'attacking_finishing',
                         'league_id', 'skill_ball_control', 'skill_long_passing', 'attacking_volleys',
                         'club_team_id', 'attacking_crossing']
        
        results = test_model_in_batches(best_model, new_data, new_data['overall'], train_columns, scaler=scaler)
        
        st.write(f"Mean Squared Error on new data: {results['mean_squared_error']:.2f}")
        st.write(f"R2 Score on new data: {results['r2_score']:.2f}")

        st.success("Model evaluation on new data is complete!")
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload a CSV file to test the model.")
