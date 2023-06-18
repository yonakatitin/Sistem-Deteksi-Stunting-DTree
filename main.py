import streamlit as st 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 

st.write("""
    # Sistem Deteksi Stunting dengan Decision Tree
""")

st.image('./banner.png')

st.write("""
    ##### Peraturan Menteri Kesehatan tentang Standar Antropometri Anak 
    https://peraturan.bpk.go.id/Home/Details/152505/permenkes-no-2-tahun-2020
""")

st.write( """
    ## Keterangan Data Yang Diinputkan
    1. jk: Jenis kelamin\n
        0 = perempuan\n
        1 = laki-laki
    2. umur: Umur\n
        1 = 0-10 bulan\n
        2 = 11-20 bulan\n
        3 = 21-30 bulan\n
        4 = 31-40 bulan\n
        5 = 41-50 bulan\n
        6 = 51-60 bulan
    3. bb/u: Berat Badan terhadap Umur\n
        1 = normal\n
        2 = kurang\n
        3 = risiko bb lebih\n
        4 = lebih
    4. tb/u: Tinggi Badan terhadap Umur\n
        1 = normal\n
        2 = pendek\n
        3 = sangat pendek

""")

#importing libraries

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('stunting1.csv')

# menentukan parameter x dan y
x = df.drop(['bb/tb'], axis = 'columns')
y = df['bb/tb']

# train test split
X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.1, random_state=1)

from sklearn.preprocessing import StandardScaler 

ss_train_test = StandardScaler()

X_train_ss_scaled = ss_train_test.fit_transform(X_train)
X_test_ss_scaled = ss_train_test.transform(X_test)

# decision tree classifier
m = DecisionTreeClassifier()
m.fit(X_train_ss_scaled, y_train)
y_pred_dt = m.predict(X_test)

st.write("# Masukkan data")


form = st.form(key='my-form')
inputGender = form.number_input("Jenis kelamin (1 = laki-laki, 0 = perempuan): ", 0)
inputAge = form.number_input("Umur (1 = 0-10 bln, 2 = 11-20 bln, 3 = 21-30 bln, 4 = 31-40 bln, 5 = 41-50 bln, 6 = 51-60 bln): ", 0)
inputBBU = form.number_input("Berat Badan terhadap Umur (1 = normal, 2 = kurang, 3 = risiko bb lebih, 4 = lebih): ", 0)
inputTBU = form.number_input("Tinggi Badan terhadap Umur (1 = normal, 2 = pendek, 3 = sangat pendek): ", 0)
submit = form.form_submit_button('Submit')

completeData = np.array([inputGender, inputAge, inputBBU, 
                        inputTBU]).reshape(1, -1)
scaledData = ss_train_test.transform(completeData)


st.write('Tekan Submit Untuk Melihat Hasil Prediksi')

if submit:
    prediction = m.predict(scaledData)
    if prediction == 1 :
        result = 'Gizi Baik'
    elif prediction == 2 :
        result = 'Gizi Kurang => Stunting'
    elif prediction == 3 :
        result = 'Risiko Gizi Lebih'
    elif prediction == 4 :
        result = 'Gizi Lebih'
    else:
        result = 'Obesitas'
    st.success(f"{result}")
    
