import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import warnings
import time
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="🎯 Smart Donation Prediction Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main styling - matching HTML exactly */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: #f0f4f8;
    }
    
    /* App container styling */
    .app-container {
        background-color: #ffffff;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border-radius: 1rem;
        max-width: 1200px;
        margin: 40px auto;
    }
    
    /* Sidebar styling - exact match */
    .sidebar {
        background-color: #1f2937;
        color: white;
        border-radius: 1rem 0 0 1rem;
    }
    
    .nav-button {
        padding: 1rem 1.5rem;
        cursor: pointer;
        transition: background-color 0.2s;
        border-radius: 0.5rem;
        display: flex;
        align-items: center;
        width: 100%;
        background: none;
        border: none;
        color: white;
        font-size: 1rem;
        margin: 0.25rem 0;
    }
    
    .nav-button:hover, .nav-button.active {
        background-color: #374151;
        color: #38bdf8;
    }
    
    /* Main content styling */
    .main-content {
        padding: 2.5rem;
    }
    
    /* Custom metric cards - exact colors from HTML */
    .metric-card-blue {
        background: #3b82f6;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-card-green {
        background: #10b981;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-card-yellow {
        background: #f59e0b;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    /* Prediction result styling - exact match */
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .prediction-success {
        background: #10b981;
        color: white;
    }
    
    .prediction-failure {
        background: #6b7280;
        color: white;
    }
    
    /* Custom buttons - exact styling */
    .stButton > button {
        background: #0ea5e9;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        background: #0284c7;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }
    
    /* Input form styling */
    .stSelectbox > div > div {
        background-color: #1f2937; /* --- التعديل هنا: تغيير اللون الأبيض إلى أزرق غامق --- */
        color: white; /* --- التعديل هنا: تغيير لون الخط إلى أبيض --- */
        border: 1px solid #374151; /* --- التعديل هنا: تغيير لون الإطار ليتناسب --- */
        border-radius: 0.375rem;
    }
    
    .stNumberInput > div > div > input {
        background-color: #1f2937; /* --- التعديل هنا: تغيير اللون الأبيض إلى أزرق غامق --- */
        color: white; /* --- التعديل هنا: تغيير لون الخط إلى أبيض --- */
        border: 1px solid #374151; /* --- التعديل هنا: تغيير لون الإطار ليتناسب --- */
        border-radius: 0.375rem;
    }
    
    /* Quick actions styling */
    .quick-actions {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Form styling */
    .form-container {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Result display styling */
    .result-container {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 2px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedDonationDashboard:
    def __init__(self):
        """Initialize the advanced dashboard."""
        self.df = None
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.predictions = None
        self.accuracy = 0.0
        self.roc_auc = 0.0
        
        self.setup_data()
        
    def setup_data(self):
        """Setup real census data and train the model."""
        # 1. Read the actual CSV
        try:
            df = pd.read_csv('C:/ami_ai/ohhh/census.csv')
        except FileNotFoundError:
            st.error("Error: census.csv file not found. Please ensure the file is uploaded.")
            return

        # 2. Data Cleaning: Strip leading/trailing whitespace from all string columns
        # This is CRITICAL for the census dataset
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

        # 3. Target Encoding: Create the binary 'donation' target
        # Income >50K is the target (1), <=50K is non-target (0)
        df['donation'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

        # 4. Feature Engineering for Visualizations
        bins = [17, 30, 45, 60, 90]
        labels = ['Youth (17-29)', 'Adult (30-44)', 'Middle-Aged (45-59)', 'Senior (60+)']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
        
        self.df = df.copy()

        # --- Model Training Preparation ---
        
        # Define features needed for the model (based on the original code's structure)
        model_features_list = [
            'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 
            # Categorical dummy features to be created
            'workclass_Federal-gov', 
            'marital-status_Married-civ-spouse', 
            'occupation_Exec-managerial', 
            'relationship_Husband', 
            'race_White', 
            'sex_Male', 
            'native-country_United-States'
        ]

        # Use pd.get_dummies for one-hot encoding on the necessary categorical columns
        categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        df_model = pd.get_dummies(df, columns=categorical_cols, drop_first=False, prefix_sep='_')
        
        # Ensure only the specific dummy features used in the original model exist (0 or 1)
        X = df_model[['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]
        
        for feature in model_features_list:
            if feature not in X.columns and feature not in ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
                if feature in df_model.columns:
                    X[feature] = df_model[feature]
                else:
                    # Add the feature column with 0s if it's missing (e.g. if a category is missing)
                    X[feature] = 0

        # Reorder columns to match the model features list exactly
        X = X[model_features_list]
        y = df_model['donation'] # Target variable

        # 5. Model Training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Store results
        self.predictions = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'probability': y_pred_proba
        })
        
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.accuracy = accuracy_score(y_test, y_pred)
        self.roc_auc = roc_auc_score(y_test, y_pred_proba)

    # --- Plotting Functions (Updated to use real data columns) ---

    def create_age_income_line_chart(self):
        """Create professional line chart showing income distribution trends by age."""
        age_income_data = self.df.groupby(['age', 'income']).size().unstack(fill_value=0)
        age_income_data['total'] = age_income_data.sum(axis=1)
        age_income_data['high_income_pct'] = (age_income_data.get('>50K', 0) / age_income_data['total'] * 100).fillna(0)
        
        fig = px.line(
            age_income_data.reset_index(),
            x='age',
            y='high_income_pct',
            title='High Income Percentage by Age',
            labels={'age': 'Age', 'high_income_pct': 'High Income Percentage (%)'}
        )
        fig.update_traces(mode='lines+markers', marker={'size': 4})
        fig.update_layout(height=400)
        return fig

    def create_education_bar_chart(self):
        """Create professional horizontal bar chart for income by education."""
        income_by_education = pd.crosstab(self.df['education_level'], self.df['income'])
        income_by_education['total'] = income_by_education.sum(axis=1)
        income_by_education['high_income_pct'] = (income_by_education.get('>50K', 0) / income_by_education['total'] * 100).fillna(0)
        
        # Sort by high income percentage
        sorted_data = income_by_education.reset_index().sort_values('high_income_pct', ascending=False)
        
        fig = px.bar(
            sorted_data,
            x='high_income_pct',
            y='education_level',
            orientation='h',
            title='High Income % by Education Level',
            color='high_income_pct',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        return fig
        
    def create_race_stacked_bar_chart(self):
        """Create professional stacked bar chart for income distribution by race."""
        race_income = pd.crosstab(self.df['race'], self.df['income'])
        race_income_pct = race_income.div(race_income.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            race_income_pct.reset_index(),
            x='race',
            y=['<=50K', '>50K'],
            title='Income Distribution by Race (%)',
            barmode='stack',
            color_discrete_map={'<=50K': '#FF6B6B', '>50K': '#10B981'}
        )
        fig.update_layout(yaxis_title="Percentage (%)", height=400)
        return fig

    def create_feature_importance_chart(self):
        """Create feature importance chart."""
        top_features = self.feature_importance.head(10)
        
        fig = px.bar(
            top_features, 
            x='importance', 
            y='feature',
            orientation='h',
            title='Top 10 Feature Importances (Model Drivers)', 
            color='importance',
            color_continuous_scale='Teal'
        )
        
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        return fig
    
    def create_confusion_matrix(self):
        """Create confusion matrix."""
        cm = confusion_matrix(self.predictions['actual'], self.predictions['predicted'])
        
        fig = px.imshow(
            cm, 
            text_auto=True, 
            aspect="auto",
            title="Model Confusion Matrix",
            color_continuous_scale='Blues',
            labels=dict(x="Predicted Label", y="True Label"),
            x=['<=50K', '>50K'],
            y=['<=50K', '>50K']
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_smart_donation_gauge(self, value):
        """Create the Smart Donation Index gauge."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Smart Donation Index"},
            gauge = {
                'axis': {'range': [None, 90]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 40], 'color': "pink"},
                    {'range': [40, 70], 'color': "lightgreen"},
                    {'range': [70, 90], 'color': "green"}
                ],
                'threshold': {
                    'line': {"color": "red", "width": 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(height=300, font={'color': "darkblue", 'family': "Inter"})
        return fig
    
    # --- Page Renderers ---

    def render_sidebar(self):
        """Render sidebar navigation and handle state."""
        st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 1.5rem; font-weight: bold; color: #38bdf8; margin-bottom: 0.5rem;">
                🎯 CharityML
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons (Working as tabs/pages)
        def nav_button(icon, label, page_name):
            if st.sidebar.button(f"{icon} {label}", use_container_width=True, key=f"nav_{page_name}"):
                st.session_state.page = page_name
                st.rerun()

        nav_button("🏠", "Dashboard", "🏠 Dashboard")
        nav_button("💸", "Predict Income", "💸 Predict Income")
        nav_button("📊", "Visualize Insights", "📊 Visualize Insights")
        nav_button("⚙", "Train Model", "🔧 Train Model")
        
        st.sidebar.markdown("---")
        
        # Quick stats in sidebar
        st.sidebar.markdown("### 📈 Quick Stats")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Records", f"{len(self.df):,}")
        with col2:
            st.metric("Accuracy", f"{self.accuracy*100:.1f}%")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align: center; color: #9ca3af; font-size: 0.8rem;">
            ML Engineer Nanodegree Project
        </div>
        """, unsafe_allow_html=True)
        
        return st.session_state.get('page', "🏠 Dashboard")

    def render_dashboard(self):
        """Render the main dashboard page."""
        st.markdown("""
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="color: #1f2937; font-size: 2rem; font-weight: 700; margin: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">
                🎯 Finding Donors – Income Predictor
            </h1>
            <p style="color: #6b7280; font-size: 1rem; margin: 1rem 0 2rem 0;">
                Welcome to the predictive modeling application for CharityML. Use this tool to predict donor eligibility and analyze census data insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card-blue">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">👥</div>
                <div style="font-size: 2rem; font-weight: bold; margin: 0;">{:,}</div>
                <div style="font-size: 0.875rem; margin: 0.5rem 0 0 0;">Total Records Processed</div>
            </div>
            """.format(len(self.df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card-green">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏆</div>
                <div style="font-size: 2rem; font-weight: bold; margin: 0;">{:.1f}%</div>
                <div style="font-size: 0.875rem; margin: 0.5rem 0 0 0;">Optimized Accuracy</div>
            </div>
            """.format(self.accuracy*100), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card-yellow">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💰</div>
                <div style="font-size: 2rem; font-weight: bold; margin: 0;">$50K+</div>
                <div style="font-size: 0.875rem; margin: 0.5rem 0 0 0;">Target Income Threshold</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("""
        <div class="quick-actions">
            <h2 style="color: #1f2937; margin-bottom: 1rem; font-size: 1.25rem;">Quick Actions</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Predict Income", use_container_width=True):
                st.session_state.page = "💸 Predict Income"
                st.rerun()
        
        with col2:
            if st.button("👁 Visualize Insights", use_container_width=True):
                st.session_state.page = "📊 Visualize Insights"
                st.rerun()
        
        with col3:
            if st.button("⚙ Train Model", use_container_width=True):
                st.info("🚀 Model training simulation completed!")
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature Importance (Moved back to the main dashboard)
        st.subheader("🔑 Top 10 Feature Importance")
        st.plotly_chart(self.create_feature_importance_chart(), use_container_width=True)
        
        # Model Performance and Distribution (Moved back to the main dashboard)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Donation Distribution")
            
            # Use the actual donation column from the real data
            donation_counts = self.df['donation'].value_counts()
            
            # Ensure names are ordered correctly
            names = ['Unlikely Donor (<=50K)', 'Likely Donor (>50K)']
            values = [donation_counts.get(0, 0), donation_counts.get(1, 0)]
            
            st.plotly_chart(px.pie(
                names=names,
                values=values,
                title='Overall Donation Distribution',
                color_discrete_sequence=['#ef4444', '#10b981']
            ), use_container_width=True)
            
        with col2:
            st.subheader("🎯 Model Performance")
            st.plotly_chart(self.create_confusion_matrix(), use_container_width=True)

    def render_prediction_page(self):
        """Render the prediction page."""
        st.markdown("""
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="color: #1f2937; font-size: 2rem; font-weight: 700; margin: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">
                💸 Predict Donor Potential
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        # Define the actual education levels and their 'education-num' mapping
        # Based on the census data, these are the common values
        education_levels_map = {
            "Doctorate": 16.0,
            "Masters": 14.0,
            "Bachelors": 13.0,
            "Assoc-acdm": 12.0,
            "Assoc-voc": 11.0,
            "Some-college": 10.0,
            "HS-grad": 9.0,
            "11th": 7.0, # Example of lower level
            "9th": 5.0 # Example of very low level
        }
        
        with col1:
            st.markdown("""
            <div class="form-container">
                <h2 style="color: #1f2937; margin-bottom: 1rem; font-size: 1.25rem;">Donor Profile Input (Features)</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Use st.form to ensure all inputs are gathered before prediction logic runs
            with st.form("prediction_form"):
                
                # --- Numerical Inputs ---
                col_age, col_edu = st.columns(2)
                age = col_age.number_input("Age", min_value=17, max_value=90, value=45, step=1, key="input_age")
                
                # Use actual keys for the selectbox, which will be mapped to 'education-num'
                education_level_display = col_edu.selectbox(
                    "Education Level", 
                    list(education_levels_map.keys()), 
                    index=2, # Default to Bachelors
                    key="input_education_level"
                )
                
                col_work, col_hours = st.columns(2)
                # Note: These are limited categories, and the model only checks for 'Federal-gov'
                workclass = col_work.selectbox(
                    "Workclass (Federal-gov indicator)",
                    self.df['workclass'].unique().tolist(),
                    key="input_workclass"
                )
                
                hours_per_week = col_hours.number_input("Hours/Week", min_value=1, max_value=99, value=40, step=1, key="input_hours_per_week")
                
                col_gain, col_loss = st.columns(2)
                capital_gain = col_gain.number_input("Capital Gain (USD)", min_value=0, max_value=100000, value=5000, step=100, key="input_capital_gain")
                capital_loss = col_loss.number_input("Capital Loss (USD)", min_value=0, max_value=10000, value=0, step=100, key="input_capital_loss")
                
                # --- Categorical Inputs (Mapped to Dummy Variables for Model) ---
                marital_status = st.selectbox(
                    "Marital Status (Married-civ-spouse indicator)", 
                    self.df['marital-status'].unique().tolist(), 
                    key="input_marital_status"
                )
                occupation = st.selectbox(
                    "Occupation (Exec-managerial indicator)", 
                    self.df['occupation'].unique().tolist(), 
                    key="input_occupation"
                )
                relationship = st.selectbox(
                    "Relationship (Husband indicator)", 
                    self.df['relationship'].unique().tolist(), 
                    key="input_relationship"
                )
                race = st.selectbox(
                    "Race (White indicator)", 
                    self.df['race'].unique().tolist(), 
                    key="input_race"
                )
                sex = st.selectbox(
                    "Sex (Male indicator)", 
                    self.df['sex'].unique().tolist(), 
                    key="input_sex"
                )
                native_country = st.selectbox(
                    "Native Country (US indicator)", 
                    self.df['native-country'].unique().tolist(), 
                    key="input_native_country"
                )
                
                submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)
                
                if submitted:
                    # Map the selected education level back to its 'education-num'
                    education_num = education_levels_map.get(education_level_display, 9.0) # Default to HS-grad if mapping fails

                    # Logic to create the required 12 feature array for the model
                    input_data = np.array([[
                        age, 
                        education_num, 
                        capital_gain, 
                        capital_loss, 
                        hours_per_week, 
                        1 if workclass == "Federal-gov" else 0, # workclass_Federal-gov
                        1 if marital_status == "Married-civ-spouse" else 0, # marital-status_Married-civ-spouse
                        1 if occupation == "Exec-managerial" else 0, # occupation_Exec-managerial
                        1 if relationship == "Husband" else 0, # relationship_Husband
                        1 if race == "White" else 0, # race_White
                        1 if sex == "Male" else 0, # sex_Male
                        1 if native_country == "United-States" else 0 # native-country_United-States
                    ]])
                    
                    input_scaled = self.scaler.transform(input_data)
                    prob = self.model.predict_proba(input_scaled)[0][1]
                    prediction = self.model.predict(input_scaled)[0]
                    
                    st.session_state.prediction_result = {
                        'probability': prob,
                        'prediction': prediction,
                        'smart_index': prob * 90
                    }
                    st.rerun() 
        
        with col2:
            st.markdown("""
            <div class="result-container">
                <h2 style="color: #1f2937; margin-bottom: 1.5rem; text-align: center; font-size: 1.25rem;">Prediction Output</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if 'prediction_result' in st.session_state:
                result = st.session_state.prediction_result
                gauge_fig = self.create_smart_donation_gauge(result['smart_index'])
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                if result['prediction'] == 1:
                    st.markdown(f"""
                    <div class="prediction-result prediction-success">
                        <div style="font-size: 1.5rem; font-weight: bold; margin: 0;">&gt; $50K (Likely Donor)</div>
                        <div style="font-size: 0.875rem; margin: 0.5rem 0 0 0;">High confidence score in donor prediction.</div>
                        <div style="margin: 0.75rem 0 0 0; font-size: 0.75rem; opacity: 0.8; display: flex; align-items: center; justify-content: center;">
                            🧪 Model Probability: {result['probability']:.4f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-result prediction-failure">
                        <div style="font-size: 1.5rem; font-weight: bold; margin: 0;">&lt;= $50K (Unlikely Donor)</div>
                        <div style="font-size: 0.875rem; margin: 0.5rem 0 0 0;">Low confidence score in donor prediction.</div>
                        <div style="margin: 0.75rem 0 0 0; font-size: 0.75rem; opacity: 0.8; display: flex; align-items: center; justify-content: center;">
                            🧪 Model Probability: {result['probability']:.4f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; color: #6b7280; padding: 2rem;">
                    <h3>Input details and click "Run Prediction"</h3>
                    <p>The Smart Donation Index will be displayed here.</p>
                </div>
                """, unsafe_allow_html=True)

    def render_visualization_page(self):
        """Render the visualization page."""
        st.markdown("""
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="color: #1f2937; font-size: 2rem; font-weight: 700; margin: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">
                📊 Visualization & Model Insights
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Row 1: Age Trend
        st.header("📈 Donation Rate Trends")
        st.plotly_chart(self.create_age_income_line_chart(), use_container_width=True)
        
        # Row 2: Demographic Analysis
        col1, col2 = st.columns(2)
        with col1:
            st.header("👨‍🎓 Income by Education")
            st.plotly_chart(self.create_education_bar_chart(), use_container_width=True)
        with col2:
            st.header("🌍 Income by Race")
            st.plotly_chart(self.create_race_stacked_bar_chart(), use_container_width=True)
        
        # Row 3: Model Performance
        st.header("🎯 Model Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            st.plotly_chart(self.create_confusion_matrix(), use_container_width=True)
        with col2:
            st.subheader("Key Metrics")
            
            # Calculate Precision and Recall on the fly for the dashboard (using the test set results)
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(self.predictions['actual'], self.predictions['predicted'])
            recall = recall_score(self.predictions['actual'], self.predictions['predicted'])
            
            st.markdown("""
                <div style="background: #f8fafc; padding: 2rem; border-radius: 1rem;">
                    <h3 style="color: #1f2937; margin-bottom: 1rem;">Model Metrics</h3>
                </div>
                """, unsafe_allow_html=True)
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Accuracy", f"{self.accuracy*100:.2f}%")
                st.metric("ROC AUC", f"{self.roc_auc:.4f}")
            with col_metric2:
                st.metric("Precision", f"{precision:.4f}")
                st.metric("Recall", f"{recall:.4f}")


    def render_train_model_page(self):
        """Render the train model page."""
        st.markdown("""
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="color: #1f2937; font-size: 2rem; font-weight: 700; margin: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">
                ⚙ Model Retraining
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1); border: 1px solid #e5e7eb;">
            <h2 style="color: #1f2937; margin-bottom: 1rem; font-size: 1.25rem;">Model Training Configuration</h2>
            <p style="color: #6b7280; margin-bottom: 1.5rem;">Configure and simulate retraining your model with advanced parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛 Model Parameters")
            algorithm = st.selectbox("Algorithm", ["Random Forest (Current)", "XGBoost", "AdaBoost", "SVM"])
            n_estimators = st.slider("Number of Estimators", 10, 200, 100)
            max_depth = st.slider("Max Depth", 3, 20, 10)
        
        with col2:
            st.markdown("### 📊 Training Configuration")
            test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
            random_state = st.number_input("Random State", value=42)
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        if st.button("🚀 Start Retraining Simulation", use_container_width=True):
            with st.spinner(f"Training {algorithm} with {n_estimators} estimators..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Simulate new metrics
                new_acc = self.accuracy + np.random.uniform(-0.02, 0.02)
                new_auc = self.roc_auc + np.random.uniform(-0.01, 0.01)
                self.accuracy = max(0.75, min(0.95, new_acc))
                self.roc_auc = max(0.75, min(0.99, new_auc))
                
                # Re-run feature importance simulation (must maintain the same features)
                new_importance = np.random.dirichlet(np.ones(len(self.feature_importance)), size=1)[0]
                self.feature_importance['importance'] = new_importance / new_importance.sum()
                self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)

                st.success("✅ Model retraining completed successfully! Metrics updated.")
                st.balloons()
                st.session_state.page = "🏠 Dashboard"
                st.rerun()

    def run_dashboard(self):
        """Run the main dashboard application."""
        # Only run if data was loaded successfully
        if self.df is not None and not self.df.empty:
            if 'page' not in st.session_state:
                st.session_state.page = "🏠 Dashboard"
            
            selected_page = self.render_sidebar()
            
            st.markdown("""<div class="app-container">""", unsafe_allow_html=True)
            st.markdown("""<div class="main-content">""", unsafe_allow_html=True)
            
            if selected_page == "🏠 Dashboard":
                self.render_dashboard()
            elif selected_page == "💸 Predict Income":
                self.render_prediction_page()
            elif selected_page == "📊 Visualize Insights":
                self.render_visualization_page()
            elif selected_page == "🔧 Train Model":
                self.render_train_model_page()

            st.markdown("""</div>""", unsafe_allow_html=True) # End main-content
            st.markdown("""</div>""", unsafe_allow_html=True) # End app-container
        else:
            st.error("Cannot run the dashboard. Data loading failed.")

def main():
    """Main function to run the dashboard."""
    dashboard = AdvancedDonationDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()