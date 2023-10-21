import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(
    page_title= 'Churn Score',
    layout='wide',
    initial_sidebar_state = 'expanded'
    )

def run ():
    # Add Title
    st.title('Churn Score Prediction')

    # Add Sub Header
    st.subheader('EDA of Dataset')


    # Add Description

    st.write('Made by **Abi Sugiri**')
    st.write('FTDS RMT 021')
    
    

    # Adding lines
    st.markdown('----')

    # Show DataFrame
    #Loading Data
    data = pd.read_csv('churn.csv')
    st.dataframe(data)

    # Membuat Heatmap
    st.write('#### Plot Correlation Heatmap')
    df_heatmap = data.filter(["age", "days_since_last_login", "avg_time_spent", "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet", "churn_risk_score"], axis=1)
    fig = plt.figure(figsize=(15, 5))
    sns.heatmap(df_heatmap.corr(),annot=True)
    st.pyplot(fig)

    # Membuat Histogram berdasarkan input User
    st.write('#### Pie Chart based on user input')
    option = st.selectbox('Choose Column : ', ('avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet'))
    fig = plt.figure(figsize=(15, 5))
    sns.histplot(data[option], bins=30, kde=True)
    st.pyplot(fig)

    # Membuat Scatter plot berdasarkan input User
    st.write('#### Scatter plot based on user input')
    option_x = st.selectbox('Choose X : ', ('avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet'))
    option_y = st.selectbox('Choose Y : ', ('avg_transaction_value', 'avg_time_spent', 'avg_frequency_login_days', 'points_in_wallet'))
    option_hue = st.selectbox('Choose Hue : ', ('gender', 'region_category', 'membership_category', 'joined_through_referral', 'used_special_discount'))
    fig = plt.figure(figsize=(15, 5))
    sns.scatterplot(data, x=data[option_x], y=data[option_y], hue=data[option_hue])
    st.pyplot(fig)


if __name__ == '__main__':
    run()