import streamlit as st
import requests
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Page Setup
st.set_page_config(page_title="📦 Perishable Goods Prediction")

# Custom Heading
st.markdown("<h1 style='text-align: left; color: #4B8BBE;'> Perishable Goods Sales Predictor</h1>", unsafe_allow_html=True)
st.markdown("### Forecast sales and optimize inventory for perishable items.")

# User Input
with st.form("form"):
    cold_storage = st.number_input("Cold Storage Capacity", value=500)
    shelf_life = st.number_input("Shelf Life Days", value=3, min_value=1)
    marketing_spend = st.number_input("Marketing Spend", value=670.37, min_value=0.0)
    product_id = st.number_input("Product ID", value=1, min_value=1)
    store_id = st.number_input("Store ID", value=1, min_value=1)
    week_number = st.text_input("Week Number", value='2024-W01')
    wastage_unit = st.number_input("Wastage units", value=100, min_value=0)
    product_category = st.selectbox("Product Category", ['Bakery', 'Meat', 'Beverages', 'Dairy'])
    product_name = st.text_input("Product Name", value="Whole Wheat Bread 800g")
    price = st.number_input("Price", value=2.5, min_value=0.0)
    rainfall = st.number_input("Rainfall", value=20.5)
    avg_temp = st.number_input("Average Temperature", value=22.3)
    region = st.selectbox("Region", ['London', 'Midlands', 'North East', 'North West', 'South East', 'South West'])
    store_size = st.number_input("Store Size", value=1500)

    # Form submit button
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        data = {
            "Cold_Storage_Capacity": cold_storage,
            "Shelf_Life_Days": shelf_life,
            "Marketing_Spend": marketing_spend,
            "Product_ID": product_id,
            "Store_ID": store_id,
            "Week_Number": week_number,
            "Wastage_Units": wastage_unit,
            "Product_Category": product_category,
            "Product_Name": product_name,
            "Price": price,
            "Rainfall": rainfall,
            "Avg_Temperature": avg_temp,
            "Region": region,
            "Store_Size": store_size
        }

        api_url = "http://localhost:8000/predict"  
        response = requests.post(url=api_url, json={"records": [data]})

        if response.status_code == 200:
            result = response.json()
            result = result.get("predictions")
            st.write(f"Estimated Unit Sold: {int(result[0])}")
        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error(str(e))

# --- Branding / Footer ---
st.markdown("""
<div style="text-align: center;">
    <img src="https://cdn-icons-png.flaticon.com/512/2204/2204445.png" width="100">
    <p style="font-size: 16px; color: gray; margin-top: 5px;">
        Optimizing<br>Perishable<br>Goods Sales
    </p>
</div>
""", unsafe_allow_html=True)

# --- Historical Chart Section ---
st.markdown("---")
st.subheader("📈 Historical Sales Trend")

merged_data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), '..', 'model', 'merged_data.csv')
)

# Convert 'Month' column properly to datetime
merged_data['Month'] = pd.to_datetime(merged_data['Month'], errors='coerce')  

# Drop NaT if any due to coercion
merged_data = merged_data.dropna(subset=['Month'])

# Product selection 
selected_product = st.selectbox("Select Product", merged_data['Product_Name'].unique())
filtered_data = merged_data[merged_data['Product_Name'] == selected_product]

# Prepare initial monthly aggregated data
monthly_sales = merged_data.groupby(merged_data['Month'].dt.to_period('M'))['Units_Sold'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()

# Add Rolling Average (3-month trend)
monthly_sales['Rolling_Avg'] = monthly_sales['Units_Sold'].rolling(window=3, min_periods=1).mean()

# Streamlit Date Slider for dynamic filtering
min_date = merged_data['Month'].min().date() 
max_date = merged_data['Month'].max().date() 

# Date range slider
date_range = st.slider(
    "Select Month Range", 
    min_value=min_date, 
    max_value=max_date, 
    value=(min_date, max_date), 
    format="YYYY-MM")

# Compute rolling average on monthly_sales
monthly_sales['Rolling_Avg'] = (
    monthly_sales['Units_Sold']
    .rolling(window=3, min_periods=1)
    .mean()
)

# Filter from monthly_sales (not merged_data!)
filtered_sales = monthly_sales[(monthly_sales['Month'].dt.date >= date_range[0]) &
                               (monthly_sales['Month'].dt.date <= date_range[1])]

# Sample plot using matplotlib 
fig, ax = plt.subplots(figsize=(10, 5))

# Main monthly sales line
ax.plot(filtered_sales['Month'], filtered_sales['Units_Sold'], label='Monthly Sales', marker='o')
# 3-month rolling average trend line
ax.plot(filtered_sales['Month'], filtered_sales['Rolling_Avg'], label='3-Month Trend', linestyle='--', color='red')

ax.set_title(f" Sales Trend for Product: {selected_product}")
ax.set_xlabel("Month")
ax.set_ylabel("Units Sold")
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot
st.pyplot(fig)


if __name__ == "__main__":
    pass
