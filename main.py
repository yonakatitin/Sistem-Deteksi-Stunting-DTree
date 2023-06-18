import streamlit as st 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 

st.write("""
    # Prediksi Stunting
""")

st.image('./1A.jpg')

st.write( """
    ## Keterangan Data Yang Diinputkan
    1. Gender: Jenis kelamin
    2. Age: Umur
    3. Hypertension:
       0 = tidak memimiliki hipertensi
       1 = memiliki hipertensi
    4. Heart Disease:
       0 = tidak memiliki hipertensi
       1 = memiliki hipertensi
    5. Average Glucose Level: Rata-rata kadar glukosa dalam darah
    6. BMI: Body mass index

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

st.write("Dengan Menggunakan Decision Tree Nilai Akurasinya Adalah:")
st.write(accuracy_score(y_test, y_pred_dt))

st.write("# Masukkan data")


form = st.form(key='my-form')
inputGender = form.number_input("Jenis kelamin (1 = male, 0 = female): ", 0)
inputAge = form.number_input("Umur: ", 0)
inputBBU = form.number_input("BB/U (1 = normal, 2 = kurang, 3 = risiko bb lebih, 4 = lebih): ", 0)
inputTBU = form.number_input("TB/U (1 = normal, 2 = pendek, 3 = sangat pendek): ", 0)
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
        result = 'Gizi Kurang'
    elif prediction == 3 :
        result = 'Risiko Gizi Lebih'
    elif prediction == 4 :
        result = 'Gizi Lebih'
    else:
        result = 'Obesitas'
    st.write(result)
    
