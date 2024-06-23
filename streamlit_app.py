import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from io import BytesIO
import base64

# Chargement des données
data_path = '/Users/hugo/Desktop/Capstone IE/loan/loan_balanced_6040.csv'  
data = pd.read_csv(data_path)

# Prétraitement des données
X = data[['annual_inc', 'term', 'loan_amnt', 'home_ownership_OWN']]
y = data['loan_status']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Configuration du modèle RandomForest avec GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Régression linéaire pour prédiction des taux d'intérêt
X_interest = data[['loan_amnt', 'open_acc', 'delinq_2yrs', 'term']]
y_interest = data['int_rate']
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_interest, y_interest)

# Standardisation des données pour KMeans clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['annual_inc', 'loan_amnt']])
kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(data_scaled)

# Calcul des probabilités de défaut
data['probability_of_default'] = model.predict_proba(data[['annual_inc', 'term', 'loan_amnt', 'home_ownership_OWN']])[:, 1]
data.sort_values(by='probability_of_default', ascending=False, inplace=True)

# Application Streamlit
st.title('Loan Risk Management Dashboard')

tab1, tab2, tab3, tab4 = st.tabs(['Main Page', 'Background Information', 'New Client Default Prediction', 'Client Risk Segmentation'])

with tab1:
    st.write("""
    This dashboard helps a US loan mortgage company identify and manage at-risk clients. Using machine learning models and statistical analysis, it predicts loan defaults and provides actionable insights. Amid rising US mortgage delinquency rates due to economic uncertainty, this tool enables early identification of potential defaults and better management of at-risk clients, ensuring financial stability and improved loan portfolio management.
    """)

with tab2:
    st.header('Background Information')
    st.write('Explore various graphs that describe our dataset, which underpins the predictive tools used in the following tabs. Gain insights into loan distributions, income levels, interest rates, and more.')
    option = st.selectbox('Choose a chart:', ['Correlation Heatmap', 'Distribution of Loan Status', 'Distribution of Loan Amounts', 'Distribution of Annual Incomes', 'Distribution of Interest Rates'])

    if option == 'Correlation Heatmap':
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(fig)
    # Ajoutez d'autres conditions pour les graphiques ici selon les options.

with tab3:
    st.header('New Client Default Prediction')
    with st.form("prediction_form"):
        annual_income = st.number_input('Annual Income', min_value=0, max_value=1000000, value=120000)
        loan_term = st.number_input('Loan Term (months)', min_value=1, max_value=360, value=36)
        loan_amount = st.number_input('Loan Amount', min_value=0, max_value=1000000, value=300000)
        home_ownership = st.selectbox('Home Ownership', ['Own', 'Rent'])
        submitted = st.form_submit_button("Predict Loan Status")
        if submitted:
            home_ownership_own = 1 if home_ownership == 'Own' else 0
            input_df = pd.DataFrame({
                'annual_inc': [annual_income],
                'term': [loan_term],
                'loan_amnt': [loan_amount],
                'home_ownership_OWN': [home_ownership_own]
            })
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            st.write('Prediction:', 'Loan Denied' if prediction[0] == 1 else 'Loan Accepted')
            st.write('Probability of Default:', f"{prediction_proba[0][1] * 100:.2f}%")

with tab4:
    st.header('Client Risk Segmentation')
    # Visualisation du heatmap
    fig, ax = plt.subplots()
    sns.heatmap(data.pivot_table(values='loan_status', index='loan_amnt', columns='annual_inc', aggfunc='mean'), annot=True, cmap='RdYlGn_r')
    st.pyplot(fig)

st.sidebar.header("Loan Amount Selector")
loan_amount_selector = st.sidebar.slider('Select Loan Amount', min_value=int(data['loan_amnt'].min()), max_value=int(data['loan_amnt'].max()), value=50000)
st.sidebar.write(f'Selected Loan Amount: ${loan_amount_selector}')
