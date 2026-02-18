#!/usr/bin/env python3
"""
Analytics Dashboard - Monitor app usage and health
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analytics import analytics
from datetime import datetime, timedelta
import psutil
import socket
import time

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Analytics Dashboard")
st.markdown("Monitor application usage, health, and performance")

# Sidebar filters
with st.sidebar:
    st.markdown("### 📅 Filters")
    days = st.slider("Days to analyze", 1, 90, 7)
    refresh_interval = st.selectbox(
        "Auto-refresh interval",
        ["Disabled", "30 seconds", "1 minute", "5 minutes"]
    )

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", 
    "🏥 Health", 
    "📊 Statistics", 
    "❌ Errors", 
    "⚙️ System"
])

# TAB 1: Overview
with tab1:
    st.markdown("### Real-time Overview")
    
    # Get metrics
    sessions = analytics.get_sessions_count(days)
    classifications = analytics.get_classifications_count(days)
    errors = analytics.get_errors_count(days)
    accuracy = analytics.get_accuracy_stats(days)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "👥 Sessions",
            sessions,
            f"last {days}d"
        )
    
    with col2:
        st.metric(
            "🖼️ Classifications",
            classifications,
            f"last {days}d"
        )
    
    with col3:
        st.metric(
            "❌ Errors",
            errors,
            f"last {days}d"
        )
    
    with col4:
        st.metric(
            "🎯 Accuracy",
            f"{accuracy['accuracy']:.1f}%",
            f"{accuracy['correct']}/{accuracy['total']} correct"
        )
    
    with col5:
        if sessions > 0:
            avg_class_per_session = classifications / sessions
        else:
            avg_class_per_session = 0
        st.metric(
            "📷 Avg/Session",
            f"{avg_class_per_session:.2f}",
            "classifications"
        )
    
    st.markdown("---")
    
    # Daily trends
    st.markdown("### Daily Activity Trend")
    daily_stats = analytics.get_daily_stats(days)
    
    if len(daily_stats) > 0:
        fig = px.line(
            daily_stats,
            x="date",
            y="sessions",
            title="Sessions per Day",
            markers=True,
            line_shape="linear"
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available yet")
    
    # Classification distribution
    st.markdown("### Classification Type Distribution")
    type_dist = analytics.get_classification_type_distribution(days)
    
    if len(type_dist) > 0:
        fig = px.bar(
            type_dist,
            x="predicted_type",
            y="count",
            title="Classifications by Type",
            labels={"predicted_type": "Type", "count": "Count"},
            color="count",
            color_continuous_scale="viridis"
        )
        fig.update_xaxes(title_text="Bristol Scale Type (1-7)")
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Health
with tab2:
    st.markdown("### Application Health Status")
    
    health = analytics.get_latest_health_check()
    
    if health:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🟢" if health['status'] == "healthy" else "🔴"
            st.metric("Status", f"{status_color} {health['status'].upper()}")
        
        with col2:
            model_status = "✅" if health['model_loaded'] else "❌"
            st.metric("Model Loaded", model_status)
        
        with col3:
            db_status = "✅" if health['database_ok'] else "❌"
            st.metric("Database OK", db_status)
        
        with col4:
            st.metric("Response Time", f"{health['response_time_ms']:.0f}ms")
        
        st.markdown("---")
        st.markdown(f"**Last check:** {health['timestamp']}")
        st.markdown(f"**Disk space:** {health['disk_space_mb']:.1f} MB available")
    else:
        st.warning("No health check data available yet")
    
    st.markdown("---")
    st.markdown("### System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("CPU Usage", f"{cpu_percent}%")
    
    with col2:
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent}%")
    
    with col3:
        disk = psutil.disk_usage('/')
        st.metric("Disk Usage", f"{disk.percent}%")
    
    # Detailed system info
    st.markdown("### Detailed System Info")
    system_info = {
        "CPU Cores": psutil.cpu_count(),
        "Total Memory (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
        "Available Memory (GB)": round(psutil.virtual_memory().available / (1024**3), 2),
        "Total Disk (GB)": round(psutil.disk_usage('/').total / (1024**3), 2),
        "Free Disk (GB)": round(psutil.disk_usage('/').free / (1024**3), 2),
        "Hostname": socket.gethostname()
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")

# TAB 3: Statistics
with tab3:
    st.markdown("### Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Accuracy Analysis")
        accuracy = analytics.get_accuracy_stats(days)
        
        if accuracy['total'] > 0:
            stats_data = {
                "Metric": ["Correct", "Incorrect"],
                "Count": [accuracy['correct'], accuracy['total'] - accuracy['correct']]
            }
            stats_df = pd.DataFrame(stats_data)
            
            fig = px.pie(
                stats_df,
                values="Count",
                names="Metric",
                color_discrete_map={"Correct": "green", "Incorrect": "red"},
                title=f"Classification Accuracy ({accuracy['accuracy']:.1f}%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No accuracy data available (need user corrections)")
    
    with col2:
        st.markdown("#### Performance Metrics")
        
        # Get all classifications for stats
        conn_stats = __import__('sqlite3').connect('data/analytics/analytics.db')
        cutoff_date = datetime.now() - timedelta(days=days)
        
        stats_df = pd.read_sql_query("""
            SELECT processing_time_ms, confidence
            FROM classifications
            WHERE timestamp > ?
        """, conn_stats, params=(cutoff_date,))
        
        conn_stats.close()
        
        if len(stats_df) > 0:
            performance_metrics = {
                "Metric": [
                    "Avg Processing Time",
                    "Min Processing Time",
                    "Max Processing Time",
                    "Avg Confidence"
                ],
                "Value": [
                    f"{stats_df['processing_time_ms'].mean():.1f}ms",
                    f"{stats_df['processing_time_ms'].min():.1f}ms",
                    f"{stats_df['processing_time_ms'].max():.1f}ms",
                    f"{stats_df['confidence'].mean():.1%}"
                ]
            }
            
            perf_df = pd.DataFrame(performance_metrics)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

# TAB 4: Errors
with tab4:
    st.markdown("### Error Tracking")
    
    errors = analytics.get_errors_by_type(days)
    
    if len(errors) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Errors by Type")
            fig = px.bar(
                errors,
                x="error_type",
                y="count",
                title="Error Distribution",
                labels={"error_type": "Error Type", "count": "Count"},
                color="count",
                color_continuous_scale="reds"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Error Summary")
            for idx, row in errors.iterrows():
                st.metric(row['error_type'], row['count'])
        
        # Error details
        st.markdown("---")
        st.markdown("#### Recent Errors")
        
        conn_errors = __import__('sqlite3').connect('data/analytics/analytics.db')
        cutoff_date = datetime.now() - timedelta(days=days)
        
        errors_detail = pd.read_sql_query("""
            SELECT timestamp, error_type, error_message
            FROM errors
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 20
        """, conn_errors, params=(cutoff_date,))
        
        conn_errors.close()
        
        if len(errors_detail) > 0:
            for idx, row in errors_detail.iterrows():
                with st.expander(f"**{row['error_type']}** - {row['timestamp']}"):
                    st.code(row['error_message'])
    else:
        st.success("✅ No errors in the selected period!")

# TAB 5: System
with tab5:
    st.markdown("### System & Configuration")
    
    st.markdown("#### Application Paths")
    paths_info = {
        "Data Directory": "data/",
        "Analytics DB": "data/analytics/analytics.db",
        "User Submissions": "data/user_submissions/",
        "Training Dataset": "data/bristol_stool_dataset/",
        "Model Weights": "model_weights.pth"
    }
    
    for path_name, path_value in paths_info.items():
        from pathlib import Path
        exists = Path(path_value).exists()
        status = "✅" if exists else "❌"
        st.text(f"{status} {path_name}: {path_value}")
    
    st.markdown("---")
    st.markdown("#### Database Status")
    
    # Check database size
    import os
    db_path = Path('data/analytics/analytics.db')
    if db_path.exists():
        db_size_mb = os.path.getsize(db_path) / (1024*1024)
        st.metric("Analytics Database Size", f"{db_size_mb:.2f} MB")
    
    st.markdown("---")
    st.markdown("#### Quick Actions")
    
    if st.button("🔄 Refresh Health Check"):
        st.info("Health check would be logged here")
    
    if st.button("📥 Export Analytics Data"):
        # Export data
        conn = __import__('sqlite3').connect('data/analytics/analytics.db')
        
        sessions_df = pd.read_sql_query("SELECT * FROM sessions", conn)
        classifications_df = pd.read_sql_query("SELECT * FROM classifications", conn)
        errors_df = pd.read_sql_query("SELECT * FROM errors", conn)
        
        conn.close()
        
        # Create excel file
        with pd.ExcelWriter('analytics_export.xlsx') as writer:
            sessions_df.to_excel(writer, sheet_name='Sessions', index=False)
            classifications_df.to_excel(writer, sheet_name='Classifications', index=False)
            errors_df.to_excel(writer, sheet_name='Errors', index=False)
        
        st.success("✅ Data exported to analytics_export.xlsx")

# Footer
st.markdown("---")
st.markdown("""
### 📊 Analytics Features
- **Real-time monitoring** of application usage
- **Health checks** for system resources
- **Error tracking** and categorization
- **Performance metrics** and accuracy statistics
- **System information** and resource monitoring

*Last updated: {}*
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
