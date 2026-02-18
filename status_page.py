#!/usr/bin/env python3
"""
Status Page - Simple endpoint for checking application status
Run as: streamlit run status_page.py
"""

import streamlit as st
import json
from pathlib import Path
from analytics import analytics
from datetime import datetime
import sqlite3

st.set_page_config(
    page_title="Status Page",
    page_icon="🟢",
    layout="compact"
)

# Get status from analytics
health = analytics.get_latest_health_check()

# Check basic requirements
model_exists = Path("model_weights.pth").exists()
db_exists = Path("data/analytics/analytics.db").exists()
submissions_dir_exists = Path("data/user_submissions").exists()
dataset_dir_exists = Path("data/bristol_stool_dataset").exists()

# Determine overall status
if health and health['status'] == 'healthy' and model_exists and db_exists:
    overall_status = "🟢 Operational"
    status_color = "green"
elif health and health['status'] == 'degraded':
    overall_status = "🟡 Degraded"
    status_color = "orange"
else:
    overall_status = "🔴 Down"
    status_color = "red"

# Display main status
st.markdown(f"# {overall_status}")

# Quick status metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    status_icon = "✅" if model_exists else "❌"
    st.metric("Model", status_icon)

with col2:
    status_icon = "✅" if db_exists else "❌"
    st.metric("Database", status_icon)

with col3:
    status_icon = "✅" if submissions_dir_exists else "❌"
    st.metric("User Data", status_icon)

with col4:
    status_icon = "✅" if dataset_dir_exists else "❌"
    st.metric("Training Data", status_icon)

# Detailed info
st.markdown("---")
st.markdown("### Last Check")

if health:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.text(f"**Timestamp:** {health['timestamp']}")
    
    with col2:
        st.text(f"**Response:** {health['response_time_ms']:.0f}ms")
    
    with col3:
        st.text(f"**Disk Free:** {health['disk_space_mb']:.0f}MB")
else:
    st.warning("No recent health check data")

# JSON API endpoint
st.markdown("---")
st.markdown("### JSON Status API")

status_json = {
    "status": overall_status.replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", "").lower(),
    "timestamp": datetime.now().isoformat(),
    "components": {
        "model": "operational" if model_exists else "down",
        "database": "operational" if db_exists else "down",
        "user_data": "operational" if submissions_dir_exists else "down",
        "training_data": "operational" if dataset_dir_exists else "down"
    }
}

st.json(status_json)

# Usage instructions
st.markdown("---")
st.markdown("""
### How to Use

**Direct URL:** Access this page to check status visually

**API:** Get JSON response at:
```
/status_page
```

**Health Check Script:**
```bash
python health_check.py            # Single check
python health_check.py --continuous  # Continuous monitoring
```

**Monitoring Integration:**
- Check HTTP status code
- Monitor response time
- Track API availability
- Set up alerts on status page
""")
