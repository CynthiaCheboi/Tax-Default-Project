import pandas as pd
import streamlit as st
import numpy as np
from joblib import load
import os
import lime
from lime.lime_tabular import LimeTabularExplainer
from lime import lime_tabular
from sklearn.model_selection import train_test_split


from pygwalker.api.streamlit import StreamlitRenderer

@st.cache_data
def  read_data():
    return pd.read_csv('https://raw.githubusercontent.com/CynthiaCheboi/POA_/main/taxdefaultcleaned2.csv')





def get_pyg_renderer():

    df          = read_data()
    df['payment_time'] = np.where(df['payment_time']==0, 'Late payment', 'On-time payment')
    return StreamlitRenderer(df,default_tab='data',theme_key='vega',dark='dark')



@st.cache_resource
def model_load():
    loaded_models , loaded_model_results = load('models_metadata.pkl')
    return loaded_models , loaded_model_results



#Function to help read searialized model
def predict_model(data):
    predictions = {}

    loaded_models, loaded_model_results = model_load()
    for model, result in zip(loaded_models, loaded_model_results):
        model_name = result['model_name']
        y_predict  = model.predict(data)
        y_predict_proba = model.predict_proba(data)[:, 1]  # Probability of positive class (class 1)

        predictions[model_name] = {'prediction': y_predict, 'probability': y_predict_proba[0]}

    return predictions


def model_category_using_prediction(predictions_dict,thershold):

    if predictions_dict['probability'] > float(thershold):
        return 'Late payment'
    else:
        return 'On-time payment'

def model_category_using_y_preds(y_preds):

    if y_preds == 0:
        return 'On-time payment'
    else:
        return 'Late payment'




# Create a button to download the model
def download_objects(file_path):

    with open(file_path, "rb") as f:
        model_bytes = f.read()

    st.sidebar.download_button(
        label="Click to download",
        data=model_bytes,
        file_name=os.path.basename(file_path),
        mime="application/octet-stream"
    )

def train_test(df):
    X = df.drop(columns=['payment_time'])
    y = df['payment_time']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test, X


#def predict_fn(X):
#    loaded_models, loaded_model_results = model_load()
#    for model, result in zip(loaded_models, loaded_model_results):
#        if result['model_name']=='XGBClassifier':
#            model = model.predict_proba(X)
#    return model

def predict_fn(X):
    loaded_models, loaded_model_results = model_load()
    predictions = None
    for model, result in zip(loaded_models, loaded_model_results):
        if result['model_name'] == 'XGBClassifier':
            predictions = model.predict_proba(X)
            break
    if predictions is None:
        raise ValueError("XGBClassifier model not found.")
    return predictions


def lime_explainer(df,instance_index):

    X_train, X_test, y_train, y_test, X = train_test(df)

    explainer = lime_tabular.LimeTabularExplainer(training_data = np.array(X_train),
                                                  feature_names=X.columns.tolist(),
                                                  class_names=['On-time payment', 'Late payment'],
                                                  mode='classification',
                                                  random_state=42)
    
    instance   = X_test.iloc[[int(instance_index)]]
    explanation = explainer.explain_instance(instance.values[0], predict_fn, num_features=len(X.columns))
    html        = explanation.as_html()

    return html