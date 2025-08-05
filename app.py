import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Prediksi Solubilitas Obat", layout="centered")
st.title("🧪 Prediksi Solubilitas Obat dalam Campuran Pelarut")

st.markdown("""
Aplikasi ini memprediksi **fraksi mol obat (solubilitas)** berdasarkan parameter fisikokimia 
obat dan pelarut, menggunakan model machine learning (XGBoost).
Silakan masukkan nilai-nilai parameter di bawah ini:
""")

with st.form("prediction_form"):
    st.subheader("📅 Input Parameter")

    Drug_Molar_Mass = st.number_input("Massa Molar Obat", value=300.0)
    Drug_Solubility_Parameter = st.number_input("Parameter Solubilitas Obat", value=20.0)
    Fusion_Temperature_K = st.number_input("Temperatur Fusi (K)", value=350.0)
    Fusion_Enthalpy_kJmol = st.number_input("Entalpi Fusi (kJ/mol)", value=10.0)
    Drug_Molar_Volume = st.number_input("Volume Molar Obat", value=150.0)

    Solvent1_Molar_Mass = st.number_input("Massa Molar Solven 1", value=60.0)
    Solvent2_Molar_Mass = st.number_input("Massa Molar Solven 2", value=70.0)
    Solvent1_PS = st.number_input("PS Solven 1", value=18.0)
    Solvent2_PS = st.number_input("PS Solven 2", value=19.0)
    Solvent1_Molar_Volume = st.number_input("Volume Molar Solven 1", value=120.0)
    Solvent2_Molar_Volume = st.number_input("Volume Molar Solven 2", value=130.0)

    x1 = st.number_input("x1 (Fraksi Mol Solven 1)", value=0.5)
    x2 = st.number_input("x2 (Fraksi Mol Solven 2)", value=0.5)
    Study_Temperature = st.number_input("Temperatur Studi", value=298.15)

    submitted = st.form_submit_button("🔮 Prediksi")

if submitted:
    try:
        # Load model
        model = joblib.load("xgboost_for_solubility.pkl")

        # Buat DataFrame dari input (tanpa x3)
        input_data = pd.DataFrame([{
            'Drug_Molar_Mass': Drug_Molar_Mass,
            'Drug_Solubility_Parameter': Drug_Solubility_Parameter,
            'Fusion_Temperature_K': Fusion_Temperature_K,
            'Fusion_Enthalpy_kJmol': Fusion_Enthalpy_kJmol,
            'Drug_Molar_Volume': Drug_Molar_Volume,
            'Solvent1_Molar_Mass': Solvent1_Molar_Mass,
            'Solvent2_Molar_Mass': Solvent2_Molar_Mass,
            'Solvent1_PS': Solvent1_PS,
            'Solvent2_PS': Solvent2_PS,
            'Solvent1_Molar_Volume': Solvent1_Molar_Volume,
            'Solvent2_Molar_Volume': Solvent2_Molar_Volume,
            'x1': x1,
            'x2': x2,
            'Study_Temperature': Study_Temperature
        }])

        prediction = model.predict(input_data)[0]

        st.subheader("📊 Ringkasan Input:")
        st.dataframe(input_data)

        st.subheader("🔎 Hasil Prediksi:")
        st.metric(label="Prediksi Solubilitas (x3)", value=f"{prediction:.6f}")

    except FileNotFoundError:
        st.error("❌ Model tidak ditemukan. Pastikan file 'xgboost_for_solubility.pkl' ada.")
    except Exception as e:
        st.error(f"❌ Terjadi error saat prediksi: {e}")
