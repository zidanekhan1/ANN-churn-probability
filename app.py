import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

from tensorflow.keras.models import load_model
model = load_model('model.h5')


with open('gender_encode.pkl','rb') as file:
    gender_encode = pickle.load(file)
with open('onehot_encode_geo.pkl','rb') as file:
    onehot_encode_geo = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.title("Customer churn prediction")

geography = st.selectbox('Geography',onehot_encode_geo.categories_[0])
gender = st.selectbox('Gender',gender_encode.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [gender_encode.transform([gender])[0]],
    "Age" : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]

})
geo_encoded = onehot_encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encode_geo.get_feature_names_out(["Geography"]))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_probab = prediction[0][0]
st.write(f'Churn probability: {prediction[0][0]:.2f}')

if prediction_probab > 0.5:
    st.write("The Customer is likely to churn")
else:
    st.write("Customer is not likely to churn")