import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Solar Energy Production Predictor",
    page_icon="☀️",
    layout="centered"
)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================
MODEL_PATH = "best_solar_model.pkl"  # Update this path if necessary


@st.cache_resource
def load_model(path):
  try:
    model = joblib.load(path)
    return model
  except Exception as e:
    return None


model = load_model(MODEL_PATH)

# ==========================================
# 3. FEATURE ENGINEERING FUNCTION
# ==========================================
def feature_engg_function(df_input):
  df_loc = df_input.copy()

  # ---- date features ----
  d = pd.to_datetime(df_loc["Interconnection Date"], errors="coerce")
  df_loc["Day_Of_week"] = d.dt.dayofweek
  df_loc["Interconnection Year"] = d.dt.year
  df_loc["Interconnection Month"] = d.dt.month

  # ---- energy storage: binary flag ----
  df_loc["Has_Energy_Storage"] = (
      df_loc["Energy Storage System Size (kWac)"].notna().astype(int)
  )

  # ---- system size features ----
  dc = df_loc["Estimated PV System Size (kWdc)"]
  ac = df_loc["PV System Size (kWac)"]
  df_loc["DC_AC_Ratio"] = (dc / ac.replace(0, np.nan)).replace(
      [np.inf, -np.inf], np.nan
  )

  # ---- drop raw columns ----
  df_loc = df_loc.drop(
      columns=[
          "Interconnection Date",
          "Energy Storage System Size (kWac)",
          "Estimated PV System Size (kWdc)",
          "PV System Size (kWac)",
      ],
      errors="ignore",
  )
  return df_loc


# ==========================================
# 4. STREAMLIT USER INTERFACE
# ==========================================
st.title("☀️ Annual Solar Energy Production Predictor", text_alignment="center")


if model is None:
  st.error(
      f"⚠️ Model file not found at `{MODEL_PATH}`. Please make sure"
      " `best_solar_model.pkl` is in the correct directory."
  )
else:
  with st.form("solar_form"):
    st.subheader("📍 Location & Utility Details", text_alignment="center")
    col1, col2 = st.columns(2)

    with col1:
      utility = st.selectbox("Utility", ["Con Ed", "PSEG Long Island", "NYSEG", "ORU", "Central Hudson", "National Grid", "Rochester Gas and Electric"])
      city_town = st.text_input("City/Town", value="Brooklyn")
      county = st.text_input("County", value="Kings")

    with col2:
      zip_code = st.text_input("Zip Code", value="11225")
      division = st.text_input("Division", value="CENY-BK")
      substation = st.text_input("Substation", value="Bensonhurst_2")

    st.subheader("🛠️ Technical & Project Details", text_alignment="center")
    col3, col4 = st.columns(2)

    with col3:
      developer = st.text_input("Developer", value="SUNCO")
      metering_method = st.selectbox("Metering Method", ["NM", "RNM", "CDG",  "FIT", "RC", "RFP", "Missing"])
     
      interconnection_date = st.date_input(
          "Interconnection Date", value=datetime(2023, 12, 29)
      )

    with col4:
      est_pv_size = st.number_input(
          "Estimated PV System Size (kWdc)",
          min_value=0.0,
          value=3.05,
          step=0.1,
      )
      pv_size_ac = st.number_input(
          "PV System Size (kWac)", min_value=0.0, value=2.61, step=0.1
      )
      has_storage = st.checkbox("Has Energy Storage System?")

      storage_size = None
      if has_storage:
        storage_size = st.number_input(
            "Energy Storage System Size (kWac)",
            min_value=0.0,
            value=5.0,
            step=0.1,
        )

    submit_button = st.form_submit_button(label="Predict Annual Production")

  # ==========================================
  # 5. PREDICTION LOGIC
  # ==========================================
  if submit_button:
    
    if not (zip_code.isdigit() and len(zip_code) == 5):
            st.warning("Zip must be exactly 5 digits.")
            st.stop()
      
    # Build input DataFrame matching the raw columns expected before feature engineering
    
    input_data = pd.DataFrame({
        "Utility": [utility],
        "City/Town": [city_town],
        "County": [county],
        "Zip": [str(zip_code).zfill(5)],
        "Division": [division],
        "Substation": [substation],
        "Developer": [developer],
        "Metering Method": [metering_method],
        "Interconnection Date": [pd.to_datetime(interconnection_date)],
        "Estimated PV System Size (kWdc)": [est_pv_size],
        "PV System Size (kWac)": [pv_size_ac],
        "Energy Storage System Size (kWac)": [
            storage_size if has_storage else np.nan
        ],
    })

    # Apply feature engineering function
    try:
      processed_input = feature_engg_function(input_data)

      # Make prediction (TransformedTargetRegressor automatically handles expm1 inverse transformation)
      raw_prediction = model.predict(processed_input)[0]
      final_prediction = max(0.0, raw_prediction)  # Clip negative values to 0

      st.success("Prediction complete!")
      st.metric(
          label="Estimated Annual PV Energy Production",
          value=f"{final_prediction:,.2f} kWh",
      )

    except Exception as e:
      st.error(f"An error occurred during prediction: {e}")