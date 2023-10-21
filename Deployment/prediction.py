import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf


#Load all files

with open('final_pipeline.pkl', 'rb') as file_1:
  model_pipeline = pickle.load(file_1)

model_ann = load_model("churn_model.h5", compile=False)


def run():
    #  Form
    with st.form(key='Churn Score Prediction'):
      age = st.number_input('age', min_value=18, max_value=99, value=18, step=1, help='Age')
      gender = st.selectbox('gender', ('F', 'M'), index=1)
      region_category = st.selectbox('region_category', ('City', 'Village', 'Town'), index=1)
      membership_category = st.selectbox('membership_category', ('No Membership', 'Basic Membership', 'Silver Membership','Premium Membership', 'Gold Membership', 'Platinum Membership'), index=1)
      joined_through_referral = st.selectbox('joined_through_referral', ('Yes', 'No'), index=1)
      st.markdown('---')

      preferred_offer_types = st.selectbox('preferred_offer_types', ('Without Offers', 'Credit/Debit Card Offers', 'Gift Vouchers/Coupons'), index=1)
      medium_of_operation = st.selectbox('medium_of_operation', ('Desktop', 'Smartphone', 'Both'), index=1)
      internet_option = st.selectbox('internet_option', ('Wi-Fi', 'Fiber_Optic', 'Mobile_Data'), index=1)
      st.markdown('---')
      
      days_since_last_login = st.slider('days_since_last_login', 0, 25, 5)  
      avg_time_spent = st.number_input('avg_time_spent', min_value=0.0, max_value=3235.5, value=100.0, step=1.0)
      avg_transaction_value = st.slider('avg_transaction_value', 900, 99900, 5000)
      avg_frequency_login_days = st.slider('avg_frequency_login_days', 0.0, 70.0, 25.0)
      points_in_wallet = st.number_input('points_in_wallet', min_value=0.0, max_value=3235.0, value=100.0, step=1.0)
      st.markdown('---')

      used_special_discount = st.selectbox('used_special_discount', ('Yes', 'No'), index=1)
      offer_application_preference = st.selectbox('offer_application_preference', ('Yes', 'No'), index=1)
      past_complaint = st.selectbox('past_complaint', ('Yes', 'No'), index=1)
      complaint_status = st.selectbox('complaint_status', ('No Information Available', 'Not Applicable', 'Unsolved', 'Solved', 'Solved in Follow-up'), index=1)
      feedback = st.selectbox('feedback', ('Poor Website', 'Poor Customer Service', 'Too many ads', 'Poor Product Quality', 'No reason specified', 'Products always in Stock', 'Reasonable Price', 'Quality Customer Care', 'User Friendly Website'), index=1)
      
      
      submitted = st.form_submit_button('Predict')
      data_inf = {
      'age': age,
      'gender': gender,
      'region_category': region_category,
      'membership_category': membership_category,
      'joined_through_referral': joined_through_referral,
      'preferred_offer_types': preferred_offer_types,
      'medium_of_operation': medium_of_operation,
      'internet_option': internet_option,
      'days_since_last_login': days_since_last_login,
      'avg_time_spent': avg_time_spent,
      'avg_transaction_value': avg_transaction_value,
      'avg_frequency_login_days': avg_frequency_login_days,
      'points_in_wallet': points_in_wallet,
      'used_special_discount': used_special_discount,
      'offer_application_preference': offer_application_preference,
      'past_complaint': past_complaint,
      'complaint_status': complaint_status,
      'feedback': 'Poor Customer Service'
    }
        
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:

        #Pipeline
        data_inf_transform = model_pipeline.transform(data_inf)
        
        
        #Predict using ANN
        y_pred_inf = model_ann.predict(data_inf_transform)
        y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
        
        st.write('# Churn Risk Score : ', str(int(y_pred_inf)))

if __name__ == '__main__':
    run()