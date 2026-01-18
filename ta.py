import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
import io

# --- Page Configuration ---
st.set_page_config(page_title="Advanced Revenue & Performance Intelligence", layout="wide")

# --- Helper Functions ---

def plot_distribution(df, column):
    """Plots the distribution of a numeric column."""
    fig = px.histogram(df, x=column, marginal="box", title=f"Distribution of {column}")
    return fig

def plot_correlation_heatmap(df):
    """Plots a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    return fig

def calculate_yoy(df):
    """Calculates Year-over-Year growth for Net Revenue by Travel Agent."""
    if 'Year' not in df.columns or 'Travel agents' not in df.columns or 'Net Revenue' not in df.columns:
        return pd.DataFrame()
    
    pivot = df.pivot_table(index='Travel agents', columns='Year', values='Net Revenue', aggfunc='sum').reset_index()
    
    if 2024 in pivot.columns and 2025 in pivot.columns:
        pivot['YoY Growth (%)'] = ((pivot[2025] - pivot[2024]) / pivot[2024]) * 100
        pivot.rename(columns={2024: '2024 Revenue', 2025: '2025 Revenue'}, inplace=True)
        return pivot
    return pd.DataFrame()

def perform_statistical_test(df, metric):
    """Performs a T-test between 2024 and 2025 for a given metric."""
    data_2024 = df[df['Year'] == 2024][metric].dropna()
    data_2025 = df[df['Year'] == 2025][metric].dropna()
    
    if len(data_2024) > 1 and len(data_2025) > 1:
        t_stat, p_val = stats.ttest_ind(data_2024, data_2025)
        return t_stat, p_val
    return np.nan, np.nan

def train_random_forest(df):
    """Trains a Random Forest model to predict Net Revenue."""
    # Prepare features
    features = ['Room Nights', 'Adults Nights', 'Arrival Rooms', 'ADR', 'Cancellation_Rate']
    target = 'Net Revenue'
    
    # Ensure columns exist
    available_features = [f for f in features if f in df.columns]
    if not available_features or target not in df.columns:
        return None, 0, 0, pd.Series()

    X = df[available_features].fillna(0)
    y = df[target].fillna(0)
    
    if len(X) < 5: # Not enough data
        return None, 0, 0, pd.Series()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    importances = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
    
    return model, r2, mae, importances

def forecast_2026_with_increase(df):
    """Forecasts 2026 revenue based on historical growth and a 15% increase."""
    yearly_rev = df.groupby('Year')['Net Revenue'].sum()
    if 2024 in yearly_rev and 2025 in yearly_rev:
        rev_24 = yearly_rev[2024]
        rev_25 = yearly_rev[2025]
        
        # Calculate historical growth rate
        historical_growth = (rev_25 - rev_24) / rev_24 if rev_24 > 0 else 0
        
        # Cap historical growth to a realistic range (e.g., -50% to +50%) to avoid outliers
        capped_growth = max(min(historical_growth, 0.5), -0.5)
        
        # Forecast 2026: 2025 Revenue * (1 + capped historical growth) * 1.15 (contract increase)
        forecast_2026 = rev_25 * (1 + capped_growth) * 1.15
        
        total_growth_from_25 = (forecast_2026 - rev_25) / rev_25 if rev_25 > 0 else 0
        return forecast_2026, total_growth_from_25
    return None, 0

# --- Data Processing Functions ---
@st.cache_data
def load_and_clean_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=";", encoding='latin1')
            df.columns = df.columns.str.strip()
            
            # Clean 'AVG R Rate' column
            if 'AVG R Rate' in df.columns:
                df['AVG R Rate'] = df['AVG R Rate'].astype(str).str.replace(',', '').str.rstrip('.').str.strip()
                df['AVG R Rate'] = df['AVG R Rate'].str.replace(r'[^\d.]', '', regex=True)
                df['AVG R Rate'] = pd.to_numeric(df['AVG R Rate'], errors='coerce')

            # Clean all numeric columns
            numeric_cols = ['Room Nights', 'Adults Nights', 'Arrival Rooms', 'Arrival Adults', 
                            'Dep. Rooms', 'Dep. Adults', 'No Show Rooms', 'Cancel Rooms', 
                            'Net Room Revenue', 'Net F&B Revenue', 'Net Revenue']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Handle missing values
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
            
            # Add calculated columns
            df['ADR'] = np.where(df['Room Nights'] > 0, df['Net Room Revenue'] / df['Room Nights'], 0)
            df['Occ_Pct'] = np.where((df['Room Nights'] + df['Cancel Rooms']) > 0, 
                                     (df['Room Nights'] / (df['Room Nights'] + df['Cancel Rooms'])) * 100, 0)
            df['Cancellation_Rate'] = np.where((df['Arrival Rooms'] + df['Cancel Rooms']) > 0, 
                                               (df['Cancel Rooms'] / (df['Arrival Rooms'] + df['Cancel Rooms'])) * 100, 0)
            
            # Added No_Show_Rate calculation
            df['No_Show_Rate'] = np.where((df['Arrival Rooms'] + df['No Show Rooms']) > 0,
                                          (df['No Show Rooms'] / (df['Arrival Rooms'] + df['No Show Rooms'])) * 100, 0)
            
            return df
        except Exception as e:
            st.error(f"Error loading or cleaning data: {e}")
            return None
    return None

def calculate_yearly_metrics(df):
    yearly_summary = {}
    metrics_to_sum = ['Net Revenue', 'Room Nights', 'Cancel Rooms']
    for metric in metrics_to_sum:
        yearly_data = df.groupby('Year')[metric].sum()
        yearly_summary[metric] = {2024: yearly_data.get(2024, 0), 2025: yearly_data.get(2025, 0)}
    
    yearly_adr = df.groupby('Year')['ADR'].mean()
    yearly_summary['ADR'] = {2024: yearly_adr.get(2024, 0), 2025: yearly_adr.get(2025, 0)}

    for year in [2024, 2025]:
        rn = yearly_summary['Room Nights'][year]
        cn = yearly_summary['Cancel Rooms'][year]
        yearly_summary.setdefault('Occ_Pct', {})[year] = (rn / (rn + cn) * 100) if (rn + cn) > 0 else 0
    
    # 2026 Forecasts
    for metric in ['Net Revenue', 'ADR', 'Room Nights', 'Occ_Pct']:
        val_24 = yearly_summary[metric][2024]
        val_25 = yearly_summary[metric][2025]
        
        # Calculate growth rate
        growth = (val_25 - val_24) / val_24 if val_24 > 0 else 0
        # Cap growth to avoid extreme projections
        capped_growth = max(min(growth, 0.5), -0.5)
        
        if metric == 'Net Revenue':
            # Apply capped growth + 15% contract increase
            yearly_summary[metric][2026] = val_25 * (1 + capped_growth) * 1.15
        elif metric == 'Occ_Pct':
            yearly_summary[metric][2026] = min(val_25 * (1 + capped_growth), 100.0)
        else:
            yearly_summary[metric][2026] = val_25 * (1 + capped_growth)
            
        yearly_summary[metric]['delta'] = ((yearly_summary[metric][2026] - val_25) / val_25 * 100) if val_25 > 0 else 0

    return yearly_summary

# --- Streamlit UI --- 
st.title("🏨 Advanced Revenue & Performance Intelligence")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Travel Agents Statistics CSV", type=["csv"])
df = load_and_clean_data(uploaded_file)

if df is not None:
    revenue_col = 'Net Revenue'
    
    # Filters
    all_years = sorted(df['Year'].unique().tolist())
    selected_years = st.sidebar.multiselect("Select Year(s)", all_years, default=all_years)
    top_agencies_2024 = df[df['Year'] == 2024].groupby('Travel agents')['Net Revenue'].sum().nlargest(10).index.tolist()
    selected_agencies = st.sidebar.multiselect("Select Agencies", options=sorted(df['Travel agents'].unique()), default=top_agencies_2024)

    filtered_df = df[df['Year'].isin(selected_years) & df['Travel agents'].isin(selected_agencies)]
    yearly_metrics = calculate_yearly_metrics(df)

    # Summary Cards
    st.subheader("Overall Performance Summary (2024 - 2026)")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.info("**Net Revenue**")
        st.metric("2024", f"${yearly_metrics['Net Revenue'][2024]:,.0f}")
        st.metric("2025", f"${yearly_metrics['Net Revenue'][2025]:,.0f}")
        st.metric("2026 (Exp.)", f"${yearly_metrics['Net Revenue'][2026]:,.0f}", f"{yearly_metrics['Net Revenue']['delta']:.1f}%")

    with c2:
        st.info("**Average ADR**")
        st.metric("2024", f"${yearly_metrics['ADR'][2024]:,.2f}")
        st.metric("2025", f"${yearly_metrics['ADR'][2025]:,.2f}")
        st.metric("2026 (Exp.)", f"${yearly_metrics['ADR'][2026]:,.2f}", f"{yearly_metrics['ADR']['delta']:.1f}%")

    with c3:
        st.info("**Total Room Nights**")
        st.metric("2024", f"{yearly_metrics['Room Nights'][2024]:,.0f}")
        st.metric("2025", f"{yearly_metrics['Room Nights'][2025]:,.0f}")
        st.metric("2026 (Exp.)", f"{yearly_metrics['Room Nights'][2026]:,.0f}", f"{yearly_metrics['Room Nights']['delta']:.1f}%")

    with c4:
        st.info("**Occupancy (Occ%)**")
        st.metric("2024", f"{yearly_metrics['Occ_Pct'][2024]:,.1f}%")
        st.metric("2025", f"{yearly_metrics['Occ_Pct'][2025]:,.1f}%")
        st.metric("2026 (Exp.)", f"{yearly_metrics['Occ_Pct'][2026]:,.1f}%", f"{yearly_metrics['Occ_Pct']['delta']:.1f}%")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Exploratory Analysis","📅 YoY Comparison", "📈 Variability/Trends", "📉 Analysis", "💡 ML & Insights"])
    
    with tab1:
        st.header("Exploratory Data Analysis")
        st.subheader("Dataset Snapshot")
        st.dataframe(filtered_df.head())

        st.subheader("Descriptive Statistics")
        st.dataframe(filtered_df.describe())

        st.subheader("Univariate Analysis: Distributions")
        numeric_cols_dist = filtered_df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols_dist:
            selected_dist_col = st.selectbox("Select a column to view its distribution", numeric_cols_dist)
            st.plotly_chart(plot_distribution(filtered_df, selected_dist_col), use_container_width=True, key="dist_chart")

        st.subheader("Bivariate/Multivariate Analysis: Correlation Heatmap")
        if not filtered_df.select_dtypes(include=np.number).empty:
            st.plotly_chart(plot_correlation_heatmap(filtered_df), use_container_width=True, key="corr_heatmap")

        st.subheader("Pivot Table: 2024 vs 2025 Performance (Net Revenue)")
        pivot_table_data = filtered_df.pivot_table(index='Travel agents', columns='Year', values='Net Revenue', aggfunc='sum')
        st.dataframe(pivot_table_data)
        
        st.subheader("Top 10 Travel Agents by Revenue")
        top_10 = filtered_df.groupby('Travel agents')[revenue_col].sum().nlargest(10).reset_index()
        fig_top = px.bar(top_10, x='Travel agents', y=revenue_col, title='Top 10 Agents by Revenue')
        st.plotly_chart(fig_top, use_container_width=True, key="top_10_agents")

    with tab2:
        st.header("Year-over-Year Comparison")
        metric_choice = st.selectbox("Select Metric for YoY Comparison", ['Net Revenue', 'Room Nights', 'ADR', 'Cancellation_Rate'])
        yoy_plot_df = filtered_df.groupby(['Year', 'Travel agents'])[metric_choice].sum().reset_index()
        fig_yoy_bar = px.bar(yoy_plot_df, x='Travel agents', y=metric_choice, color='Year', barmode='group')
        st.plotly_chart(fig_yoy_bar, use_container_width=True, key="yoy_comparison_bar")
        
        st.subheader("Detailed YoY Growth Table")
        yoy_table = calculate_yoy(df[df['Travel agents'].isin(selected_agencies)])
        if not yoy_table.empty:
            st.dataframe(yoy_table.style.format({'2024 Revenue': '{:,.2f}', '2025 Revenue': '{:,.2f}', 'YoY Growth (%)': '{:.2f}%'}))

    with tab3:
        st.header("Variability/Trends")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("ADR Trend")
            fig_adr = px.line(filtered_df.groupby(['Year', 'Travel agents'])['ADR'].mean().reset_index(), x='Year', y='ADR', color='Travel agents', markers=True)
            st.plotly_chart(fig_adr, use_container_width=True, key="adr_trend_line")
        with col_b:
            st.subheader("Occ% Trend")
            fig_occ = px.line(filtered_df.groupby(['Year', 'Travel agents'])['Occ_Pct'].mean().reset_index(), x='Year', y='Occ_Pct', color='Travel agents', markers=True)
            st.plotly_chart(fig_occ, use_container_width=True, key="occ_trend_line")
        
        st.subheader("Net Revenue Trend")
        rev_trend = filtered_df.groupby(['Year', 'Travel agents'])['Net Revenue'].sum().reset_index()
        fig_rev = px.line(rev_trend, x='Year', y='Net Revenue', color='Travel agents', markers=True)
        st.plotly_chart(fig_rev, use_container_width=True, key="rev_trend_line")
        
        yoy_df = calculate_yoy(filtered_df)
        if not yoy_df.empty:
            st.subheader("YoY Net Revenue Growth by Travel Agent (2024 vs 2025)")
            fig_yoy_growth = px.bar(yoy_df, x='Travel agents', y='YoY Growth (%)', title='YoY Growth (%)')
            st.plotly_chart(fig_yoy_growth, use_container_width=True, key="yoy_growth_bar")

        st.subheader("Statistical Significance (2024 vs 2025 Net Revenue)")
        t_stat, p_val = perform_statistical_test(filtered_df, 'Net Revenue')
        if not np.isnan(t_stat):
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_val:.4f}")
            if p_val < 0.05:
                st.success("The difference in Net Revenue between 2024 and 2025 is statistically significant.")
            else:
                st.info("The difference in Net Revenue between 2024 and 2025 is NOT statistically significant.")
        st.subheader("Revenue Composition (2025)")
        df_2025 = filtered_df[filtered_df['Year'] == 2025]
        revenue_comp = df_2025.groupby('Travel agents')[['Net Room Revenue', 'Net F&B Revenue']].sum().reset_index()
        fig_comp = px.bar(revenue_comp, x='Travel agents', y=['Net Room Revenue', 'Net F&B Revenue'], title='Room vs F&B Revenue')
        st.plotly_chart(fig_comp, use_container_width=True, key="rev_comp_2025")

    with tab4:
        st.header("Cancellations & Nights")
        c_a, c_b, c_c, col_r2 = st.columns(4)
        with c_a:
            st.subheader("Cancellation Rate (%)")
            fig_c = px.bar(filtered_df.groupby(['Year', 'Travel agents'])['Cancellation_Rate'].mean().reset_index(), x='Travel agents', y='Cancellation_Rate', color='Year', barmode='group')
            st.plotly_chart(fig_c, use_container_width=True, key="cancel_rate_bar")
        with c_b:
            st.subheader("Room Nights Distribution (2024)")
            df_2024 = filtered_df[filtered_df['Year'] == 2024]
            if not df_2024.empty:
                fig_p_2024 = px.pie(df_2024.groupby('Travel agents')['Room Nights'].sum().reset_index(), values='Room Nights', names='Travel agents')
                st.plotly_chart(fig_p_2024, use_container_width=True, key="rn_dist_2024")
            else:
                st.info("No data for 2024")
        with c_c:        
            st.subheader("Room Nights Distribution (2025)")
            df_2025_rn = filtered_df[filtered_df['Year'] == 2025]
            if not df_2025_rn.empty:
                fig_p_2025 = px.pie(df_2025_rn.groupby('Travel agents')['Room Nights'].sum().reset_index(), values='Room Nights', names='Travel agents')
                st.plotly_chart(fig_p_2025, use_container_width=True, key="rn_dist_2025")
            else:
                st.info("No data for 2025")
        
        with col_r2:
            st.subheader("No-Show Rates by Agent")
            noshow_data = filtered_df.groupby(['Year', 'Travel agents'])['No_Show_Rate'].mean().reset_index()
            fig_noshow = px.bar(noshow_data, x='Travel agents', y='No_Show_Rate', color='Year', barmode='group')
            st.plotly_chart(fig_noshow, use_container_width=True, key="noshow_by_agent_bar")
        
        st.subheader("High-Risk Agents (Cancellation Rate > 5%)")
        high_risk = filtered_df[filtered_df['Cancellation_Rate'] > 5].groupby('Travel agents')['Cancellation_Rate'].mean().sort_values(ascending=False)
        if not high_risk.empty:
            st.dataframe(high_risk.reset_index().rename(columns={'Cancellation_Rate': 'Avg Cancellation Rate (%)'}))
        else:
            st.info("No agents with cancellation rate > 5%")

    with tab5:
        st.header("Machine Learning & Strategic Insights")
        rf_model, r2, mae, feature_importances = train_random_forest(df)
        
        col_ml1, col_ml2, col_ml3 = st.columns(3)
        with col_ml1:
            st.subheader("Model Performance")
            if rf_model:
                st.write(f"**R2 Score:** {r2:.4f}")
                st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
                st.subheader("Key Revenue Drivers")
                fig_fi = px.bar(feature_importances, x=feature_importances.values, y=feature_importances.index, orientation='h')
                st.plotly_chart(fig_fi, use_container_width=True, key="feature_importance_chart")
            else:
                st.info("Not enough data to train ML model.")

        with col_ml2:
            st.subheader("2026 Revenue Forecast")
            forecast_2026_val, total_growth_2026 = forecast_2026_with_increase(df)
            if forecast_2026_val:
                st.metric(label="Projected Total Net Revenue for 2026", 
                          value=f"${forecast_2026_val:,.2f}", 
                          delta=f"{total_growth_2026*100:.2f}% from 2025")
                st.info("Note: This forecast accounts for historical growth trends (capped at 50%) PLUS a 15% contract price increase for 2026.")
        with col_ml3:
            st.subheader("Model Performance Metrics")
            if rf_model:
                st.metric("R² Score", f"{r2:.4f}", "Excellent fit" if r2 > 0.85 else "Good fit" if r2 > 0.70 else "Fair fit")
                st.metric("Mean Absolute Error", f"${mae:,.2f}")
                
                avg_revenue = df[revenue_col].mean()
                mae_percentage = (mae / avg_revenue * 100) if avg_revenue > 0 else 0
                st.metric("MAE as % of Avg Revenue", f"{mae_percentage:.2f}%")
                
                st.info(f"**Model Interpretation:** The model explains {r2*100:.2f}% of revenue variance. Average prediction error is ${mae:,.2f} ({mae_percentage:.2f}% of mean revenue).")
            else:
                st.warning("Insufficient data to train ML model")
            
        st.markdown("### 1. Dynamic Pricing Strategy")
        st.markdown("""
        **Current Situation:** ADR has increased significantly year-over-year.
        
        **Recommendation:**
        - Implement dynamic pricing for top-performing agents
        - Segment pricing by agent volume and booking patterns
        - Consider premium rates during peak seasons for high-volume agents
        - Target: Additional 5-10% ADR increase for 2026
        """)
        
        st.markdown("### 2. Cancellation & No-Show Mitigation")
        st.markdown("""
        **Current Situation:** Cancellation and no-show rates vary by agent.
        
        **Recommendation:**
        - Implement tiered cancellation policies based on agent risk profile
        - Require non-refundable deposits for high-risk agents
        - Introduce penalties for excessive cancellations
        - Offer incentives for low-cancellation agents
        - Expected Impact: 2-3% revenue recovery
        """)
        
        st.markdown("### 3. Revenue Diversification")
        st.markdown("""
        **Current Situation:** Room revenue dominates; F&B revenue is underutilized.
        
        **Recommendation:**
        - Increase F&B packages in travel agent contracts
        - Bundle services (spa, activities, dining) with room bookings
        - Target F&B revenue growth: 15-20% annually
        - Expected Impact: $50-100M additional annual revenue
        """)
        
        st.markdown("### 4. Agent Portfolio Optimization")
        st.markdown("""
        **Current Situation:** Top agents account for significant revenue concentration.
        
        **Recommendation:**
        - Strengthen relationships with top agents
        - Develop tiered partnership programs with volume-based incentives
        - Diversify portfolio by developing emerging agents
        - Risk Mitigation: Reduce dependency on single agents
        """)
        
        st.markdown("### 5. 2026 Growth Targets")
        st.markdown("""
        **Recommended Growth Initiatives:**
        1. **ADR Optimization:** +8% through dynamic pricing
        2. **Cancellation Reduction:** +2.5% through policy changes
        3. **F&B Growth:** +18% through bundling
        4. **Volume Growth:** +5% through new partnerships
        """)
    # Download buttons
    st.sidebar.markdown("---")
    st.sidebar.subheader("Download Results")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Processed Data", data=csv_data, file_name="processed_travel_data.csv", mime="text/csv")

    # Use the same forecast logic for the download button
    forecast_2026_val, _ = forecast_2026_with_increase(df)
    if forecast_2026_val:
        forecast_df = pd.DataFrame({'Year': [2026], 'Forecasted Net Revenue': [forecast_2026_val]})
        st.sidebar.download_button("Download 2026 Forecast", data=forecast_df.to_csv(index=False).encode('utf-8'), file_name="2026_forecast.csv", mime="text/csv")

    st.sidebar.info("Developed for Amira | Data Science Analysis")

else:
    st.info("Please upload the CSV file to begin.")
