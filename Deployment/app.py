import streamlit as st
import eda
import prediction

page = st.sidebar.selectbox('Choose Page : ', ('EDA', 'Predict Churn Risk Score'))

if page == 'EDA':
    eda.run()
else:
    prediction.run()