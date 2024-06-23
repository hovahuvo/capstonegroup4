import streamlit as st
import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
import dash_bootstrap_components as dbc
from dash import dash_table
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('/Users/hugo/Desktop/Capstone IE/loan/loan_balanced_6040.csv')

# Data preprocessing
X = data[['annual_inc', 'term', 'loan_amnt', 'home_ownership_OWN']]
y = data['loan_status']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Define the Random Forest Classifier with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
model = grid_search.best_estimator_

# Prepare data for linear regression to predict interest rates
X_interest = data[['loan_amnt', 'open_acc', 'delinq_2yrs', 'term']]
y_interest = data['int_rate']

# Train a Linear Regression model for predicting interest rates
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_interest, y_interest)

# Standardize data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['annual_inc', 'loan_amnt']])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(data_scaled)

# Compute default probabilities for each client using the Random Forest Classifier
data['probability_of_default'] = model.predict_proba(data[['annual_inc', 'term', 'loan_amnt', 'home_ownership_OWN']])[:, 1]
data.sort_values(by='probability_of_default', ascending=False, inplace=True)

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            dbc.Col(html.Img(src="/assets/LendSmart_logo.png", height="100px", style={"display": "block", "margin-left": "auto", "margin-right": "auto"}), width={"size": 6, "offset": 3}),
        ),
        dbc.Row(
            dbc.Col(
                dcc.Tabs(id="tabs-example", value='tab-1', children=[
                    dcc.Tab(label='Main Page', value='tab-1'),
                    dcc.Tab(label='Background Information', value='tab-2'),
                    dcc.Tab(label='New Client Default Prediction', value='tab-3'),
                    dcc.Tab(label='Client Risk Segmentation', value='tab-4'),
                ])
            )
        ),
        html.Div(id='tabs-content-example')
    ],
    style={"background-color": "#ffffff"}
)

# Define callback to render content based on selected tab
@app.callback(
    Output('tabs-content-example', 'children'),
    Input('tabs-example', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div(
                "This dashboard helps a US loan mortgage company identify and manage at-risk clients. Using machine learning models and statistical analysis, it predicts loan defaults and provides actionable insights. Amid rising US mortgage delinquency rates due to economic uncertainty (Financial Times), this tool enables early identification of potential defaults and better management of at-risk clients, ensuring financial stability and improved loan portfolio management.", 
                style={"text-align": "center", "color": "white", "padding": "20px", "border-radius": "10px", "font-size": "20px", "margin-bottom": "20px", "margin-top": "20px", "background-color": "#1B49A4"}
            ),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div(
                "Explore various graphs that describe our dataset, which underpins the predictive tools used in the following tabs. Gain insights into loan distributions, income levels, interest rates, and more.", 
                style={"text-align": "center", "color": "white", "padding": "20px", "border-radius": "10px", "font-size": "20px", "margin-bottom": "20px", "margin-top": "20px", "background-color": "#1B49A4"}
            ),
    
            dcc.Dropdown(
                id='dropdown-selection',
                options=[
                    {'label': 'Correlation Heatmap', 'value': 'heatmap'},
                    {'label': 'Distribution of Loan Status', 'value': 'loan_status'},
                    {'label': 'Distribution of Loan Amounts', 'value': 'loan_amounts'},
                    {'label': 'Distribution of Annual Incomes', 'value': 'annual_incomes'},
                    {'label': 'Distribution of Interest Rates', 'value': 'interest_rates'}
                ],
                value='heatmap'
            ),
            html.Div(id='display-selected-value', style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-top': '20px'})
        ])
    elif tab == 'tab-3':
        return dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    dbc.Col(
                        html.P("Enter your information to receive a personalized loan recommendation in seconds. Our tool quickly evaluates your eligibility, helping you save time and determine the feasibility of your loan application. If your loan is denied, you will receive a recommendation. If your loan is approved, we will suggest an interest rate.", 
                               style={"text-align": "center", "color": "white", "padding": "20px", "margin-top": "20px", "border-radius": "10px", "font-size": "20px", "margin-bottom": "20px","background-color": "#1B49A4"}),
                        width=12
                    )
                ),
                dbc.Row([
                    dbc.Col(html.Label('Annual Income', style={"color": "#2c3e50"}), width={"size": 6, "offset": 3}),
                    dbc.Col(dcc.Input(id='annual-income', type='number', value=120000, min=0, max=1000000, style={"width": "100%", "padding": "10px"}), width={"size": 6, "offset": 3})
                ]),
                dbc.Row([
                    dbc.Col(html.Label('Loan Term (months)', style={"color": "#2c3e50"}), width={"size": 6, "offset": 3}),
                    dbc.Col(dcc.Input(id='loan-term', type='number', value=36, min=1, max=360, style={"width": "100%", "padding": "10px"}), width={"size": 6, "offset": 3})
                ]),
                dbc.Row([
                    dbc.Col(html.Label('Loan Amount', style={"color": "#2c3e50"}), width={"size": 6, "offset": 3}),
                    dbc.Col(dcc.Input(id='loan-amount', type='number', value=300000, min=0, max=1000000, style={"width": "100%", "padding": "10px"}), width={"size": 6, "offset": 3})
                ]),
                dbc.Row([
                    dbc.Col(html.Label('Home Ownership (OWN=1, RENT=0)', style={"color": "#2c3e50"}), width={"size": 6, "offset": 3}),
                    dbc.Col(dcc.Input(id='home-ownership', type='number', value=1, min=0, max=1, style={"width": "100%", "padding": "10px"}), width={"size": 6, "offset": 3})
                ]),
                dbc.Row([
                    dbc.Col(html.Label('Number of Open Accounts', style={"color": "#2c3e50"}), width={"size": 6, "offset": 3}),
                    dbc.Col(dcc.Input(id='class-open-acc', type='number', value=5, min=0, max=50, style={"width": "100%", "padding": "10px"}), width={"size": 6, "offset": 3})
                ]),
                dbc.Row([
                    dbc.Col(html.Label('Delinquencies in Last 2 Years 1=YES 0=NO', style={"color": "#2c3e50"}), width={"size": 6, "offset": 3}),
                    dbc.Col(dcc.Input(id='class-delinq-2yrs', type='number', value=0, min=0, max=50, style={"width": "100%", "padding": "10px"}), width={"size": 6, "offset": 3})
                ]),
                dbc.Row(
                    dbc.Col(
                        html.Button('Predict', id='predict-button', n_clicks=0, style={"background-color": "#1B49A4", "color": "white", "padding": "20px 25px", "border-radius": "20px", "font-size": "20px"}),
                        width={"size": 6, "offset": 3},
                        style={"padding-top": "20px","display": "flex"}
                    ),
                    justify="start"
                ),
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            [
                                html.H3('Prediction', style={"color": "white"}),
                                html.Div(id='prediction-result', style={"color": "white"}),
                                html.H3('Prediction Probability', style={"color": "white"}),
                                html.Div(id='prediction-probability', style={"color": "white"}),
                                html.H3(id='recommendations-title', style={"color": "white"}),
                                html.Div(id='recommendations', style={"color": "white"})
                            ],
                            style={"padding": "20px", "backgroundColor": "#1B49A4", "borderRadius": "10px", "margin-top": "20px"}  # Adding top margin to space out results}
                        ),
                        width=6, style={"padding": "10px"}
                    ),
                    justify="center"
                )
            ]
        )
    elif tab == 'tab-4':
        fig, ax = plt.subplots(figsize=(8, 8))
        risk_levels = data.pivot_table(values='loan_status', 
                                       index=pd.cut(data['loan_amnt'], bins=range(0, 105000, 5000)), 
                                       columns=pd.cut(data['annual_inc'], bins=range(0, 1050000, 50000)), 
                                       aggfunc='mean')
        sns.heatmap(risk_levels, annot=True, cmap='RdYlGn_r', linewidths=0.5, ax=ax)
        plt.title('Client Risk Segmentation Heatmap')
        plt.xlabel('Annual Income')
        plt.ylabel('Loan Amount')
        buf = BytesIO()
        plt.savefig(buf, format="png")
        data_buf = base64.b64encode(buf.getbuffer()).decode("utf8")
        plt.close(fig)
        
        datatable = dash_table.DataTable(
            id='client-table',
            columns=[
                {"name": "Client", "id": "client"},
                {"name": "Annual Income", "id": "annual_inc"},
                {"name": "Loan Term", "id": "term"},
                {"name": "Loan Amount", "id": "loan_amnt"},
                {"name": "Home Ownership", "id": "home_ownership"},
                {"name": "Delinquencies in Last 2 Years", "id": "delinq_2yrs"},
                {"name": "Probability of Default", "id": "probability_of_default"},
                {"name": "Current Interest Rate", "id": "int_rate"},
                {"name": "Suggested Interest Rate", "id": "suggested_interest_rate"}
            ],
            data=data[data['probability_of_default'] < 1].assign(
                client=lambda x: x.index + 1,
                home_ownership=lambda x: x['home_ownership_OWN'].map({1: 'OWN', 0: 'RENT'}),
                suggested_interest_rate=lambda x: lin_reg_model.predict(
                    x[['loan_amnt', 'open_acc', 'delinq_2yrs', 'term']]
                ).round(2)
            ).to_dict('records'),
            sort_action="native",
            sort_mode="single",
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
            }
        )

        return html.Div([
            html.H4("Client Risk Segmentation Analysis", style={'textAlign': 'center', 'margin-top': '20px'}),
            html.P(
                "This heatmap visualizes the risk segmentation of clients based on their loan amounts and annual incomes. "
                "Each cell represents the default probability for a specific segment, with colors ranging from green (low risk) to red (high risk). "
                "By analyzing this heatmap, we can identify which client segments are more likely to default on their loans, allowing for better risk management and targeted strategies.",
                style={"text-align": "center", "color": "white", "backgroundColor": "#1B49A4", "padding": "10px", "border-radius": "5px", "font-size": "20px", "margin-top": "20px", "margin-bottom": "15px"}
            ),
        
            html.Div([html.Img(src='data:image/png;base64,{}'.format(data_buf))], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-top': '20px'}),
            html.Div([
                html.H4("Client Risk Evaluation and Interest Rate Recommendations", style={'textAlign': 'center', 'margin-top': '20px'}),
                html.Div("We're using our random forest model to calculate a new probability of default for all existing clients. Based on these probabilities, we've also calculated suggested interest rates. The goal is to improve the management of the company's at-risk clients.",
                         style={"text-align": "center", "color": "white", "backgroundColor": "#1B49A4", "padding": "10px", "border-radius": "5px", "font-size": "20px", "margin-top": "30px", "margin-bottom": "20px"}),
                datatable
            ], style={'padding-top': '20px'})
        ])

# Define the callback to update the displayed graph based on dropdown selection
@app.callback(
    Output('display-selected-value', 'children'),
    [Input('dropdown-selection', 'value')]
)
def update_output(value):
    if value == 'heatmap':
        correlation_matrix = data[['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 
                                   'delinq_2yrs', 'home_ownership_OWN', 'home_ownership_RENT', 'open_acc', 'loan_status']].corr()
        fig1 = plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 10},
                    xticklabels=['Loan Amount', 'Loan Term', 'Interest Rate', 'Instalment', 'Annual Income', 
                                 'Delinquency in the Last 2 Years', 'Home Owner', 'Home Renter', 'Number of Open Accounts', 'Loan Status'],
                    yticklabels=['Loan Amount', 'Loan Term', 'Interest Rate', 'Instalment', 'Annual Income', 
                                 'Delinquency in the Last 2 Years', 'Home Owner', 'Home Renter', 'Number of Open Accounts', 'Loan Status'])
        plt.title('Correlation Heatmap')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        buf1 = BytesIO()
        plt.savefig(buf1, format="png")
        data1 = base64.b64encode(buf1.getbuffer()).decode("utf8")
        plt.close(fig1)
        return html.Div(
            [html.Img(src='data:image/png;base64,{}'.format(data1))],
            style={'textAlign': 'center', 'margin-top': '20px'}
        )
    elif value == 'loan_status':
        fig2, ax2 = plt.subplots()
        data['loan_status'].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_title('Distribution of Loan Status')
        ax2.set_xlabel('Loan Status')
        ax2.set_ylabel('Number of Loans')
        buf2 = BytesIO()
        plt.savefig(buf2, format="png")
        data2 = base64.b64encode(buf2.getbuffer()).decode("utf8")
        plt.close(fig2)
        return html.Div([html.Img(src='data:image/png;base64,{}'.format(data2))])
    elif value == 'loan_amounts':
        fig3, ax3 = plt.subplots()
        data['loan_amnt'].plot(kind='hist', bins=50, color='blue', ax=ax3)
        ax3.set_title('Distribution of Loan Amounts')
        ax3.set_xlabel('Loan Amount ($)')
        buf3 = BytesIO()
        plt.savefig(buf3, format="png")
        data3 = base64.b64encode(buf3.getbuffer()).decode("utf8")
        plt.close(fig3)
        return html.Div([html.Img(src='data:image/png;base64,{}'.format(data3))])
    elif value == 'annual_incomes':
        fig4, ax4 = plt.subplots()
        data['annual_inc'].plot(kind='hist', bins=50, color='purple', ax=ax4)
        ax4.set_title('Distribution of Annual Incomes')
        ax4.set_xlabel('Annual Income ($)')
        buf4 = BytesIO()
        plt.savefig(buf4, format="png")
        data4 = base64.b64encode(buf4.getbuffer()).decode("utf8")
        plt.close(fig4)
        return html.Div([html.Img(src='data:image/png;base64,{}'.format(data4))])
    elif value == 'interest_rates':
        fig5, ax5 = plt.subplots()
        data['int_rate'].plot(kind='hist', bins=50, color='green', ax=ax5)
        ax5.set_title('Distribution of Interest Rates')
        ax5.set_xlabel('Interest Rate (%)')
        buf5 = BytesIO()
        plt.savefig(buf5, format="png")
        data5 = base64.b64encode(buf5.getbuffer()).decode("utf8")
        plt.close(fig5)
        return html.Div([html.Img(src='data:image/png;base64,{}'.format(data5))])

# Define the callback to update the predictions
@app.callback(
    [Output('prediction-result', 'children'),
     Output('prediction-probability', 'children'),
     Output('recommendations-title', 'children'),
     Output('recommendations', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('annual-income', 'value'),
     State('loan-term', 'value'),
     State('loan-amount', 'value'),
     State('home-ownership', 'value'),
     State('class-open-acc', 'value'),
     State('class-delinq-2yrs', 'value')]
)
def update_prediction(n_clicks, annual_income, loan_term, loan_amount, home_ownership, open_acc, delinq_2yrs):
    if n_clicks > 0:
        input_data = pd.DataFrame({
            'annual_inc': [annual_income],
            'term': [loan_term],
            'loan_amnt': [loan_amount],
            'home_ownership_OWN': [home_ownership]
        })

        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            result = 'Loan Denied'
            probability = f"{prediction_proba[0][1]*100:.2f}% probability of default"
            recommendations_title = 'Recommendations'
            recommendations = """
            - Reduce Loan Amount: A lower loan amount reduces the repayment burden, which can decrease the risk of default.
            - Extend Loan Term: Smaller monthly payments can be easier to manage, reducing the risk of default.
            """
            return result, probability, recommendations_title, recommendations
        else:
            result = 'Loan Accepted'
            probability = f"{prediction_proba[0][1]*100:.2f}% probability of default"

            # Predict the interest rate using the linear regression model
            input_data_for_rate = pd.DataFrame({
                'loan_amnt': [loan_amount],
                'open_acc': [open_acc],
                'delinq_2yrs': [delinq_2yrs],
                'term': [loan_term]
            })

            predicted_rate = lin_reg_model.predict(input_data_for_rate)
            recommended_rate = f"The suggested interest rate is {predicted_rate[0]:.2f}%."

            recommendations_title = 'Suggested Interest Rate'
            return result, probability, recommendations_title, recommended_rate
    return '', '', '', ''

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8197)
