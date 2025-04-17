import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# Set page config
st.set_page_config(page_title="World Happiness Dashboard", layout="wide")

# Load data
df = pd.read_csv(r"C:\Users\Palak Aswale\Desktop\Python\datasets\2024.csv")
df

# Clean column names if needed
df.columns = df.columns.str.strip()

# Sidebar filters
st.sidebar.title("Filters")
selected_year = st.sidebar.selectbox("Select Year", ["2019"])  # Assuming 2019 data
region_filter = st.sidebar.multiselect("Filter by Region", options=df['Country or region'].unique())

if region_filter:
    df = df[df['Country or region'].isin(region_filter)]

# Main content
st.title("🌍 World Happiness Report Dashboard")
st.markdown("Analyzing factors that contribute to happiness across countries")

# Tab layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Top 10 Countries", 
    "GDP vs Happiness", 
    "Regional Analysis",
    "Correlation Analysis",
    "Predictive Model"
])

# Tab 1: Top 10 Happiest Countries
with tab1:
    st.header("Top 10 Happiest Countries")
    top_10 = df.sort_values('Score', ascending=False).head(10)
    fig1 = px.bar(top_10, 
                 x='Score', 
                 y='Country or region', 
                 orientation='h',
                 color='Score',
                 color_continuous_scale='Viridis',
                 title='Top 10 Happiest Countries by Happiness Score')
    st.plotly_chart(fig1, use_container_width=True)
    
    st.dataframe(top_10[['Country or region', 'Score', 'GDP per capita', 'Social support']].style.background_gradient(cmap='Blues'), 
                use_container_width=True)

# Tab 2: GDP vs Happiness Score
with tab2:
    st.header("Relationship Between GDP and Happiness")
    
    col1, col2 = st.columns(2)
    with col1:
        log_scale = st.checkbox("Log Scale for GDP", value=False)
    with col2:
        trendline = st.selectbox("Trendline", [None, "ols", "lowess"])
    
    fig2 = px.scatter(df, 
                     x='GDP per capita', 
                     y='Score',
                     color='Country or region',
                     hover_name='Country or region',
                     trendline=trendline,
                     log_x=log_scale,
                     title='GDP per capita vs Happiness Score')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("""
    **Insights:**
    - Generally, countries with higher GDP per capita tend to have higher happiness scores
    - The relationship appears to be positive but not perfectly linear
    - Some countries achieve higher happiness scores than their GDP would predict
    """)

# Tab 3: Regional Analysis
with tab3:
    st.header("Regional Happiness Analysis")
    
    # Pie chart of happiness by region
    region_happiness = df.groupby('Country or region')['Score'].sum().reset_index()
    fig3 = px.pie(region_happiness, 
                 values='Score', 
                 names='Country or region',
                 title='Contribution to Total Happiness by Region',
                 hole=0.3)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Box plot of happiness distribution by region
    fig4 = px.box(df, 
                 x='Country or region', 
                 y='Score',
                 color='Country or region',
                 title='Happiness Score Distribution by Region')
    st.plotly_chart(fig4, use_container_width=True)

# Tab 4: Correlation Analysis
with tab4:
    st.header("Correlation Between Happiness Factors")
    
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                ax=ax)
    plt.title('Correlation Matrix of Happiness Factors')
    st.pyplot(fig)
    
    st.markdown("""
    **Key Observations:**
    - GDP per capita and Healthy life expectancy show the strongest correlation with happiness score
    - Social support is also highly correlated with happiness
    - Generosity shows the weakest correlation among all factors
    """)

# Tab 5: Predictive Modeling
with tab5:
    st.header("Happiness Score Prediction Model")
    
    st.markdown("""
    This linear regression model predicts happiness score based on:
    - GDP per capita
    - Social support
    """)
    
    # Prepare data for modeling
    X = df[['GDP per capita', 'Social support']]
    y = df['Score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    
    # Display model results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model R-squared Score", f"{r2:.3f}")
    with col2:
        st.metric("Number of Training Samples", X_train.shape[0])
    
    # Show coefficients
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    st.dataframe(coef_df, use_container_width=True)
    
    # Create actual vs predicted plot
    fig5 = px.scatter(x=y_test, y=y_pred,
                     labels={'x': 'Actual Happiness Score', 'y': 'Predicted Happiness Score'},
                     title='Actual vs Predicted Happiness Scores')
    fig5.add_shape(type='line', line=dict(dash='dash'),
                  x0=y_test.min(), y0=y_test.min(),
                  x1=y_test.max(), y1=y_test.max())
    st.plotly_chart(fig5, use_container_width=True)
    
    # Prediction interface
    st.subheader("Make Your Own Prediction")
    col1, col2 = st.columns(2)
    with col1:
        gdp = st.slider("GDP per capita", 
                        min_value=float(df['GDP per capita'].min()), 
                        max_value=float(df['GDP per capita'].max()),
                        value=float(df['GDP per capita'].mean()))
    with col2:
        social = st.slider("Social Support", 
                          min_value=float(df['Social support'].min()), 
                          max_value=float(df['Social support'].max()),
                          value=float(df['Social support'].mean()))
    
    prediction = model.predict([[gdp, social]])[0]
    st.metric("Predicted Happiness Score", f"{prediction:.2f}")

# Footer
st.markdown("---")
st.markdown("Data Source: World Happiness Report 2019 from Kaggle")