import joblib
import streamlit as st
import pandas as pd
import os

current_dir = os.path.dirname(__file__)
pipeline_path = os.path.join(current_dir, "Model", "saved_model_pipeline.pkl")

pipeline = joblib.load(pipeline_path)

pkname = st.selectbox("Select Peak's Name", ['Everest', 'Ama Dablam', 'Baruntse', 'Cho Oyu', 'Gyalzen Peak',
       'Himlung Himal', 'Kyungka Ri 2', 'Luza', 'Manaslu', 'Annapurna I',
       'Dhaulagiri I', 'Kangchung Shar', 'Lhotse', 'Makalu', 'Pumori',
       'Tilicho', 'Tengkangpoche', 'Tukuche', 'Annapurna III', 'Chamlang',
       'Chekigo', 'Chobuje', 'Cholatse', 'Dolma Khang', 'Dorje Lhakpa',
       'Gyajikang', 'Jannu', 'Kangchenjunga', 'Kangtega', 'Langdung',
       'Mariyang', 'Omitso Go', 'Omoga Ri Chang', 'Panbari',
       'Purbung Himal', 'Purkhung', 'Putha Hiunchuli', 'Rokapi',
       'Surma-Sarovar North', 'Tengkoma', 'Amphu Gyabjen', 'Annapurna IV',
       'Dorje Lakpa II', 'Gangapurna', 'Kangchung Nup', 'Nuptse East I',
       'Nuptse', 'Phu Kang', 'Pokharkang', 'Ratna Chuli', 'Saula',
       'Tengi Ragi Tau South', 'Bhemdang Ri', 'Bhrikuti Shail',
       'Chandi Himal', 'Chukyima Go', 'Chumbu', 'Chulu West',
       'Dhaulagiri II', 'Dogari', 'Ganchenpo', 'Hongku Chuli', 'Hongku',
       'Jannu East', 'Khatung Khang', 'Lamjung Himal', 'Langtang Lirung',
       'Lachama Chuli', 'Lachama North', 'Lunag Ri', 'Metalung',
       'Nagoru Far East', 'Panalotapa', 'Phungi', 'Ripimo Shar',
       'Sat Peak', 'Shershon', 'Sita Chuchura', 'Thamserku', 'Bhrikuti',
       'Goldum Peak', 'Jarkya', 'Kabru South', 'Kirat Chuli',
       'Langtang Yubra', 'Langpo South', 'Lobuche East', 'Takphu Himal',
       'Tengi Ragi Tau', 'Tutse', 'Anidesh Chuli', 'Chhopa Bamare',
       'Cho Polu', 'Drohmo', 'Hulang Go', 'Jugal 5', 'Khangri Shar',
       'Khamjung', 'Khayang', 'Makalu II', 'Malanphulan', 'Nemjung',
       'Phole', 'Phurbi Chhyachu', 'Raksha Urai', 'Rolwaling Kang',
       'Sharphu VI', 'Yansa Tsenji', 'Hungchhi', 'Jugal 1', 'Jugal 2',
       'Jugal 3', 'Kyabura', 'Lingtren', 'Patrasi Himal', 'Yalung Peak'])
o2used = st.selectbox("Was Oxygen Used?", ["Yes", "No"])
season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
nohired = st.selectbox("Hired Professionals?", ["Yes", "No"])

# season_encoding = {'Spring': 1, 'Summer': 2, 'Autumn': 3, 'Winter': 4}
# season_encoded = season_encoding.get(season, 4)


input_dict = {
    'PKNAME' : pkname,
    'O2USED' : 1 if o2used == 'Yes' else 0, 
    'SEASON_FACTOR' : season,
    'NOHIRED' : 1 if nohired == 'No' else 0
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Success"):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.write("✅ Success" if prediction == 1 else "❌ Failure")
    st.write(f"Probability of Success: {probability:.2f}")
