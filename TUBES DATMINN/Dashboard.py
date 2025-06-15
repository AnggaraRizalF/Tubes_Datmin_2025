import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle
import joblib
import os
import base64
import datetime

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

print("Loading data and models...")
try:
    df = pd.read_csv('data/preprocessed_cancer_data_clean.csv')
    print("Loaded clean preprocessed data")
except FileNotFoundError:
    try:
        df = pd.read_csv('data/preprocessed_cancer_data.csv')
        print("Clean data not found, loaded original preprocessed data")
        
        missing_values = df.isnull().sum()
        missing_cols = missing_values[missing_values > 0]
        if len(missing_cols) > 0:
            print("Missing values detected in the following columns:")
            print(missing_cols)
            
            default_values = {
                'CancerStage': 2,  
                'TreatmentType': 3,  
                'AlcoholUse': 1,  
                'TumorType': 0,  
                'Age': df['Age'].median() if 'Age' in df.columns and not df['Age'].isnull().all() else 60,
                'TumorSize': df['TumorSize'].median() if 'TumorSize' in df.columns and not df['TumorSize'].isnull().all() else 3.5,
                'ChemotherapySessions': df['ChemotherapySessions'].median() if 'ChemotherapySessions' in df.columns and not df['ChemotherapySessions'].isnull().all() else 6,
                'RadiationSessions': df['RadiationSessions'].median() if 'RadiationSessions' in df.columns and not df['RadiationSessions'].isnull().all() else 10,
                'SmokingStatus': 1,  
                'Gender': 0,  
                'Metastasis': 0, 
                'Cluster': 0  
            }

            for col in df.columns:
                if col in default_values and df[col].isnull().sum() > 0:
                    print(f"Filling {df[col].isnull().sum()} missing values in {col} with {default_values[col]}")
                    df[col] = df[col].fillna(default_values[col])
            
            df = df.fillna(0)
            print("All missing values filled")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        df = pd.DataFrame({
            'Age': np.random.normal(60, 10, 100),
            'TumorSize': np.random.normal(3.5, 1.5, 100),
            'TreatmentType': np.random.randint(1, 4, 100),
            'TumorType': np.random.randint(0, 6, 100),
            'ChemotherapySessions': np.random.randint(0, 12, 100),
            'RadiationSessions': np.random.randint(0, 20, 100),
            'CancerStage': np.random.randint(1, 5, 100),
            'SmokingStatus': np.random.randint(0, 3, 100),
            'AlcoholUse': np.random.randint(0, 3, 100),
            'Gender': np.random.randint(0, 2, 100),
            'Metastasis': np.random.randint(0, 2, 100),
            'SurvivalStatus': np.random.randint(0, 2, 100),
            'Cluster': np.random.randint(0, 3, 100)
        })
        print("Created dummy dataset for demonstration")

try:
    kmeans_model = joblib.load('model/kmeans_model.pkl')
    logistic_model = joblib.load('model/logistic_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    print("Successfully loaded all models")
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {str(e)}")
    models_loaded = False

if 'Cluster' not in df.columns:
    print("Cluster column not found, adding with random values")
    df['Cluster'] = np.random.randint(0, 3, len(df))

def get_age_distribution():
    if 'Age' in df.columns:
        fig = px.histogram(df, x='Age', color='Cluster', 
                         title='Distribusi Usia Pasien',
                         labels={'Age': 'Usia', 'count': 'Jumlah Pasien'})
        return fig
    return go.Figure()

def get_gender_distribution():
    if 'Gender' in df.columns:
        gender_counts = df.groupby('Gender').size().reset_index(name='count')
        gender_counts['Gender'] = gender_counts['Gender'].map({0: 'Perempuan', 1: 'Laki-laki'})
        fig = px.pie(gender_counts, values='count', names='Gender', 
                    title='Distribusi Jenis Kelamin')
        return fig
    return go.Figure()

def get_tumor_type_distribution():
    if 'TumorType' in df.columns:
        tumor_mapping = {
            0: 'Payudara',
            1: 'Serviks',
            2: 'Kolorektal',
            3: 'Hati',
            4: 'Paru-paru',
            5: 'Perut'
        }
        df_tumor = df.copy()
        df_tumor['TumorType'] = df_tumor['TumorType'].map(tumor_mapping)
        fig = px.bar(df_tumor.groupby('TumorType').size().reset_index(name='count'), 
                    x='TumorType', y='count',
                    title='Distribusi Jenis Tumor',
                    labels={'TumorType': 'Jenis Tumor', 'count': 'Jumlah Pasien'})
        return fig
    return go.Figure()

def get_cluster_visualization():
    if 'Cluster' in df.columns:
        try:
            features = ['Age', 'TumorType',  'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 
                        'TreatmentType', 'CancerStage', 'SmokingStatus', 'AlcoholUse', 
                        'Gender', 'Metastasis']
            
            features_to_use = [f for f in features if f in df.columns]
            
            if len(features_to_use) >= 2:  
                X = df[features_to_use].fillna(0)  
                pca = PCA(n_components=2)
                components = pca.fit_transform(X)
                
                df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                df_pca['Cluster'] = df['Cluster']
                
                fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                                title='Visualisasi Cluster dengan PCA',
                                labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
                return fig
            
        except Exception as e:
            print(f"Error creating cluster visualization: {str(e)}")
    
    fig = px.scatter(x=[0, 1, 2], y=[0, 1, 2], title="Tidak dapat membuat visualisasi cluster")
    return fig

def get_cluster_profiles():
    if 'Cluster' in df.columns:
        try:
            profile_features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 
                                'CancerStage', 'SurvivalStatus']
            
            profile_features = [f for f in profile_features if f in df.columns]
            
            if len(profile_features) > 0:
                profiles = df.groupby('Cluster')[profile_features].mean().reset_index()
                
                profiles_melted = pd.melt(profiles, id_vars=['Cluster'], 
                                        value_vars=profile_features,
                                        var_name='Feature', value_name='Average Value')
                
                fig = px.bar(profiles_melted, x='Feature', y='Average Value', color='Cluster',
                            barmode='group', title='Profil Rata-rata Cluster')
                return fig
        
        except Exception as e:
            print(f"Error creating cluster profiles: {str(e)}")
    
    fig = px.bar(x=['No Data'], y=[0], title="Tidak dapat membuat profil cluster")
    return fig

def get_confusion_matrix():
    try:
        img_path = 'results/confusion_matrix.png'
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            
            fig = html.Img(src=f'data:image/png;base64,{encoded_image}',
                          style={'width': '100%', 'max-width': '800px'})
            return fig
    except Exception as e:
        print(f"Error loading confusion matrix image: {str(e)}")
    
    return html.Div("Gambar confusion matrix tidak tersedia. Jalankan test_models.py terlebih dahulu.")

def get_roc_curve():
    try:
        img_path = 'results/roc_curve.png'
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            
            fig = html.Img(src=f'data:image/png;base64,{encoded_image}',
                          style={'width': '100%', 'max-width': '800px'})
            return fig
    except Exception as e:
        print(f"Error loading ROC curve image: {str(e)}")
    
    # Fallback to empty div
    return html.Div("Gambar ROC curve tidak tersedia. Jalankan test_models.py terlebih dahulu.")

def get_feature_importance():
    try:
        img_path = 'results/feature_importance.png'
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            
            fig = html.Img(src=f'data:image/png;base64,{encoded_image}',
                          style={'width': '100%', 'max-width': '800px'})
            return fig
    except Exception as e:
        print(f"Error loading feature importance image: {str(e)}")
    
    return html.Div("Gambar feature importance tidak tersedia. Jalankan test_models.py terlebih dahulu.")

app.layout = html.Div([
    html.H1("Dashboard Analisis Pasien Kanker", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H4("⚠️ Peringatan: Model tidak berhasil dimuat", style={'color': 'red'}),
            html.P("Beberapa fitur dashboard mungkin tidak berfungsi. Silakan jalankan save_models.py terlebih dahulu.")
        ], style={'padding': '10px', 'border': '1px solid red', 'borderRadius': '5px', 'backgroundColor': '#ffeeee'})
    ] if not models_loaded else [], id='model-warning'),
    
    dcc.Tabs([
        dcc.Tab(label='Overview', children=[
            html.Div([
                html.H3("Distribusi Demografis Pasien"),
                html.Div([
                    dcc.Graph(id='age-distribution', figure=get_age_distribution()),
                    dcc.Graph(id='gender-distribution', figure=get_gender_distribution()),
                    dcc.Graph(id='tumor-distribution', figure=get_tumor_type_distribution())
                ])
            ])
        ]),
        
        dcc.Tab(label='Exploratory Analysis', children=[
            html.Div([
                html.H3("Analisis Eksplorasi Data"),
                html.Div([
                    html.Div([
                        html.Label('Pilih Variabel X:'),
                        dcc.Dropdown(
                            id='x-variable',
                            options=[{'label': col, 'value': col} for col in df.columns],
                            value='Age' if 'Age' in df.columns else df.columns[0]
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                    html.Div([
                        html.Label('Pilih Variabel Y:'),
                        dcc.Dropdown(
                            id='y-variable',
                            options=[{'label': col, 'value': col} for col in df.columns],
                            value='TumorSize' if 'TumorSize' in df.columns else df.columns[0]
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                    html.Div([
                        html.Label('Pilih Jenis Plot:'),
                        dcc.Dropdown(
                            id='plot-type',
                            options=[
                                {'label': 'Scatter Plot', 'value': 'scatter'},
                                {'label': 'Bar Plot', 'value': 'bar'},
                                {'label': 'Box Plot', 'value': 'box'},
                                {'label': 'Histogram', 'value': 'histogram'}
                            ],
                            value='scatter'
                        )
                    ], style={'width': '30%', 'display': 'inline-block'})
                ]),
                dcc.Graph(id='exploratory-plot')
            ])
        ]),
        
        dcc.Tab(label='Clustering Results', children=[
            html.Div([
                html.H3("Hasil K-means Clustering"),
                dcc.Graph(id='cluster-visualization', figure=get_cluster_visualization()),
                dcc.Graph(id='cluster-profiles', figure=get_cluster_profiles()),
                html.Div([
                    html.Label('Pilih Cluster:'),
                    dcc.Dropdown(
                        id='cluster-selector',
                        options=[
                            {'label': 'Cluster 0: Young Survivors', 'value': 0},
                            {'label': 'Cluster 1: Mid-stage Patients', 'value': 1},
                            {'label': 'Cluster 2: Advanced Cases', 'value': 2}
                        ],
                        value=0
                    )
                ], style={'width': '50%'}),
                dcc.Graph(id='cluster-details')
            ])
        ]),
        dcc.Tab(label='Survival Prediction', children=[
            html.Div([
                html.H3("Prediksi Status Kelangsungan Hidup"),
                
                html.Div([
                    html.Div([
                        html.P("⚠️ Catatan: Model ini menggunakan data sintetis untuk menyeimbangkan kelas. Hasil prediksi hanya untuk tujuan demonstrasi.", 
                              style={'color': '#856404', 'marginBottom': '0'})
                    ], style={'padding': '10px', 'backgroundColor': '#fff3cd', 'borderRadius': '5px', 'marginBottom': '20px'})
                ] if not models_loaded else []),
                
                html.Div([
                    html.Div([
                        html.Label('Usia:'),
                        dcc.Input(id='input-age', type='number', value=60),
                        html.Label('Jenis Kelamin:'),
                        dcc.Dropdown(
                            id='input-gender',
                            options=[
                                {'label': 'Laki-laki', 'value': 1},
                                {'label': 'Perempuan', 'value': 0}
                            ],
                            value=1
                        ),
                        html.Label('Jenis Tumor:'),
                        dcc.Dropdown(
                            id='input-tumor-type',
                            options=[
                                {'label': 'Payudara', 'value': 0},
                                {'label': 'Serviks', 'value': 1},
                                {'label': 'Kolorektal', 'value': 2},
                                {'label': 'Hati', 'value': 3},
                                {'label': 'Paru-paru', 'value': 4},
                                {'label': 'Lambung', 'value': 5}
                            ],
                            value=0
                        ),
                        html.Label('Stadium Kanker:'),
                        dcc.Dropdown(
                            id='input-cancer-stage',
                            options=[
                                {'label': 'Stadium I', 'value': 3},
                                {'label': 'Stadium II', 'value': 2},
                                {'label': 'Stadium III', 'value': 1},
                                {'label': 'Stadium IV', 'value': 0}
                            ],
                            value=2
                        )
                    ], style={'width': '30%', 'float': 'left'}),
                                  
                    html.Div([
                        html.Label('Ukuran Tumor (cm):'),
                        dcc.Input(id='input-tumor-size', type='number', value=4.0),
                        html.Label('Metastasis:'),
                        dcc.Dropdown(
                            id='input-metastasis',
                            options=[
                                {'label': 'Ya', 'value': 0},
                                {'label': 'Tidak', 'value': 1}
                            ],
                            value=0
                        ),
                        html.Label('Jenis Pengobatan:'),
                        dcc.Dropdown(
                            id='input-treatment-type',
                            options=[
                                {'label': 'Operasi', 'value': 3},
                                {'label': 'Kemoterapi', 'value': 0},
                                {'label': 'Radiasi', 'value': 2},
                                {'label': 'Immunotherapy', 'value': 1},
                                {'label': 'Targeted Therapy', 'value': 4}
                            ],
                            value=3
                        ),
                        html.Label('Status Merokok:'),
                        dcc.Dropdown(
                            id='input-smoking-status',
                            options=[
                                {'label': 'Tidak Pernah', 'value': 2},
                                {'label': 'Jarang', 'value': 1},
                                {'label': 'Perokok Aktif', 'value': 0}
                            ],
                            value=1
                        ),
                        
                    ], style={'width': '30%', 'float': 'left', 'marginLeft': '5%'}),
                    html.Div([
                        html.Label('Konsumsi Alkohol:'),
                        dcc.Dropdown(
                            id='input-alcohol-use',
                            options=[
                                {'label': 'Tidak', 'value': 2},
                                {'label': 'Moderat', 'value': 1},
                                {'label': 'Berat', 'value': 0}
                            ],
                            value=1
                        ),
                        html.Label('Sesi Kemoterapi:'),
                        dcc.Input(id='input-chemo-sessions', type='number', value=5),
                        html.Label('Sesi Radiasi:'),
                        dcc.Input(id='input-radiation-sessions', type='number', value=10),
                        html.Br(),
                        html.Button('Prediksi', id='predict-button', n_clicks=0,
                                   style={'backgroundColor': '#4CAF50', 'color': 'white', 
                                          'padding': '10px 15px', 'borderRadius': '5px',
                                          'border': 'none', 'marginTop': '20px', 'cursor': 'pointer'})
                    ], style={'width': '30%', 'float': 'left', 'marginLeft': '5%'})
                ], style={'overflow': 'hidden'}),
                html.Div(id='prediction-output', style={'marginTop': '50px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px','alignItems': 'flex-start'})
        ]),
        
        dcc.Tab(label='Model Performance', children=[
            html.Div([
                html.H3("Performa Model Prediktif"),
                html.Div([
                    html.P("Untuk menampilkan visualisasi performa model, jalankan script test_models.py terlebih dahulu."),
                    html.P("File-file visualisasi akan disimpan di direktori 'results'.")
                ], style={'padding': '10px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px', 'marginBottom': '20px'}),
                
                html.Div([
                    html.H4("Confusion Matrix", style={'marginTop': '30px'}),
                    html.Div(id='confusion-matrix', children=get_confusion_matrix()),
                    
                    html.H4("ROC Curve", style={'marginTop': '30px'}),
                    html.Div(id='roc-curve', children=get_roc_curve()),
                    
                    html.H4("Feature Importance", style={'marginTop': '30px'}),
                    html.Div(id='feature-importance', children=get_feature_importance())
                ])
            ])
        ])
    ]),
    html.Div([
        html.Hr(),
        html.P(f"Dashboard Analisis Pasien Kanker | Data terakhir diupdate: {datetime.datetime.now().strftime('%Y-%m-%d')}"),
        html.P("Catatan: Dashboard ini dibuat untuk tujuan pembelajaran dan demonstrasi.")
    ], style={'marginTop': '50px', 'textAlign': 'center', 'color': '#6c757d'})
])

@app.callback(
    Output('exploratory-plot', 'figure'),
    [Input('x-variable', 'value'),
     Input('y-variable', 'value'),
     Input('plot-type', 'value')]
)
def update_exploratory_plot(x_var, y_var, plot_type):
    try:
        if x_var not in df.columns or y_var not in df.columns:
            return px.scatter(title="Error: One or both selected variables not found in dataset")
        
        temp_df = df.copy()
        if temp_df[x_var].isnull().any() or temp_df[y_var].isnull().any():
            temp_df = temp_df.fillna(0)
        
        if plot_type == 'scatter':
            fig = px.scatter(temp_df, x=x_var, y=y_var, color='Cluster',
                           title=f'Scatter Plot: {x_var} vs {y_var}')
        elif plot_type == 'bar':
            if x_var == y_var:
                fig = px.histogram(temp_df, x=x_var, color='Cluster',
                                 title=f'Histogram: {x_var}')
            else:
                if temp_df[x_var].nunique() > 10 and pd.api.types.is_numeric_dtype(temp_df[x_var]):
                    temp_df[f'{x_var}_binned'] = pd.cut(temp_df[x_var], bins=10)
                    group_var = f'{x_var}_binned'
                else:
                    group_var = x_var
                
                grouped = temp_df.groupby([group_var, 'Cluster'])[y_var].mean().reset_index()
                fig = px.bar(grouped, x=group_var, y=y_var, color='Cluster',
                           title=f'Bar Plot: {x_var} vs {y_var}')
        elif plot_type == 'box':
            fig = px.box(temp_df, x=x_var, y=y_var, color='Cluster',
                       title=f'Box Plot: {x_var} vs {y_var}')
        else: 
            fig = px.histogram(temp_df, x=x_var, color='Cluster',
                             title=f'Histogram: {x_var}')
        
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray')
        )
        
        return fig
    
    except Exception as e:
        print(f"Error in exploratory plot: {str(e)}")
        return px.scatter(title=f"Error creating plot: {str(e)}")

@app.callback(
    Output('cluster-details', 'figure'),
    [Input('cluster-selector', 'value')]
)
def update_cluster_details(selected_cluster):
    try:
        if 'Cluster' not in df.columns:
            return px.scatter(title="Error: Cluster column not found in dataset")
        
        cluster_df = df[df['Cluster'] == selected_cluster]
        
        if len(cluster_df) == 0:
            return px.scatter(title=f"No data for Cluster {selected_cluster}")
        
        if 'TumorType' in cluster_df.columns and 'SurvivalStatus' in cluster_df.columns:
            tumor_mapping = {
                0: 'Payudara',
                1: 'Cervical',
                2: 'Colorectal',
                3: 'Hati',
                4: 'Paru-paru',
                5: 'Lambung'
            }
            cluster_df['TumorType'] = cluster_df['TumorType'].map(tumor_mapping)
            
            survival_by_tumor = cluster_df.groupby('TumorType')['SurvivalStatus'].mean().reset_index()
            
            fig = px.bar(survival_by_tumor, x='TumorType', y='SurvivalStatus',
                       title=f'Tingkat Kelangsungan Hidup berdasarkan Jenis Tumor untuk Cluster {selected_cluster}',
                       labels={'TumorType': 'Jenis Tumor', 
                              'SurvivalStatus': 'Tingkat Kelangsungan Hidup'})
            
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray', range=[0, 1])
            )
            
            return fig
        else:
            if 'Age' in cluster_df.columns:
                fig = px.histogram(cluster_df, x='Age',
                                 title=f'Distribusi Usia untuk Cluster {selected_cluster}')
                return fig
            
        return px.scatter(title=f"Detail untuk Cluster {selected_cluster} tidak tersedia")
    
    except Exception as e:
        print(f"Error in cluster details: {str(e)}")
        return px.scatter(title=f"Error menampilkan detail cluster: {str(e)}")

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-age', 'value'),
     State('input-gender', 'value'),
     State('input-tumor-type', 'value'),
     State('input-cancer-stage', 'value'),
     State('input-tumor-size', 'value'),
     State('input-metastasis', 'value'),
     State('input-treatment-type', 'value'),
     State('input-smoking-status', 'value'),
     State('input-alcohol-use', 'value'),
     State('input-chemo-sessions', 'value'),
     State('input-radiation-sessions', 'value')]
)
def predict_survival(n_clicks, age, gender, tumor_type, cancer_stage, tumor_size,
                    metastasis, treatment_type, smoking_status, alcohol_use,
                    chemo_sessions, radiation_sessions):
    if n_clicks == 0:
        return html.Div()
    
    if not models_loaded:
        return html.Div([
            html.H4("Model tidak tersedia", style={'color': 'red'}),
            html.P("Tidak dapat melakukan prediksi karena model tidak berhasil dimuat. Jalankan save_models.py terlebih dahulu.")
        ], style={'padding': '20px', 'backgroundColor': '#f8d7da', 'borderRadius': '5px'})
    
    try:
        features = ['Age', 'TumorType',  'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 
                    'TreatmentType', 'CancerStage', 'SmokingStatus', 'AlcoholUse', 
                    'Gender', 'Metastasis']
        
        input_data = np.zeros(len(features))
        
        feature_values = {
            'Age': age,
            'TumorType': tumor_type,
            'TumorSize': tumor_size,
            'ChemotherapySessions': chemo_sessions,
            'RadiationSessions': radiation_sessions,
            'TreatmentType': treatment_type,
            'CancerStage': cancer_stage,
            'SmokingStatus': smoking_status,
            'AlcoholUse': alcohol_use,
            'Gender': gender,
            'Metastasis': metastasis
        }
        
        for i, feature in enumerate(features):
            input_data[i] = feature_values.get(feature, 0)
        
        input_scaled = scaler.transform([input_data])
        
        survival_prob = logistic_model.predict_proba(input_scaled)[0, 1]
        prediction = "Survived" if survival_prob >= 0.5 else "Deceased"
        prediction_id = "survived" if survival_prob >= 0.5 else "deceased"
        
        return html.Div([
            html.Div([
                html.H4(f"Prediksi Status: ", style={'display': 'inline-block', 'marginRight': '10px'}),
                html.H4(prediction, style={'display': 'inline-block', 
                                           'color': 'green' if prediction == "Survived" else 'red'})
            ]),
            html.H5(f"Probabilitas Bertahan Hidup: {survival_prob:.2f}"),
            dcc.Graph(
                figure=go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=survival_prob * 100,
                        title={'text': "Probabilitas Kelangsungan Hidup (%)"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "darkgreen"},
                               'steps': [
                                   {'range': [0, 25], 'color': "red"},
                                   {'range': [25, 50], 'color': "orange"},
                                   {'range': [50, 75], 'color': "yellow"},
                                   {'range': [75, 100], 'color': "lightgreen"}
                               ],
                               'threshold': {
                                   'line': {'color': "black", 'width': 4},
                                   'thickness': 0.75,
                                   'value': survival_prob * 100
                               }}
                    )
                )
            ),
            html.Div([
                html.H5("Faktor-faktor Penting yang Mempengaruhi Prediksi:"),
                html.Ul([
                    html.Li(f"Usia: {age} tahun", className=prediction_id),
                    html.Li(f"Stadium Kanker: {cancer_stage}", 
                           className=prediction_id if cancer_stage < 3 else "risk-factor"),
                    html.Li(f"Metastasis: {'Ya' if metastasis == 1 else 'Tidak'}", 
                           className=prediction_id if metastasis == 0 else "risk-factor"),
                    html.Li(f"Ukuran Tumor: {tumor_size} cm", 
                           className=prediction_id if tumor_size < 5 else "risk-factor"),
                ])
            ], style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
        ])
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return html.Div([
            html.H4("Error dalam Prediksi", style={'color': 'red'}),
            html.P(f"Terjadi kesalahan: {str(e)}")
        ], style={'padding': '20px', 'backgroundColor': '#f8d7da', 'borderRadius': '5px'})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)