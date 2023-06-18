import streamlit as st 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 

st.write("""
    # Prediksi Stroke
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

df = pd.read_csv('stroke.csv')

# replace data kosong
df['bmi'].fillna(df['bmi'].mean(), inplace = True)

# hapus kolom id
df.drop('id', axis = 1,inplace = True)

# handling categorical features
en = LabelEncoder()

cols = ['gender']
for col in cols:
  df[col] = en.fit_transform(df[col])

# menentukan parameter x dan y
x = df.drop(['stroke', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis = 'columns')
y = df['stroke']

# train test split
X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2, random_state=69)

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
inputHyper = form.number_input("Apakah mempunyai hipertensi? (1 = ya, 0 = tidak): ", 0)
inputHD = form.number_input("Apakah mempunyai penyakit jantung? (1 = ya, 0 = tidak): ", 0)
inputGlucose = form.number_input("Rata-rata kadar glukosa: ", 0)
inputBMI = form.number_input("BMI: ", 0)
submit = form.form_submit_button('Submit')

completeData = np.array([inputGender, inputAge, inputHyper, 
                        inputHD, inputGlucose, inputBMI]).reshape(1, -1)
scaledData = ss_train_test.transform(completeData)


st.write('Tekan Submit Untuk Melihat Hasil Prediksi')

if submit:
    prediction = m.predict(scaledData)
    if prediction == 1 :
        result = 'Stroke'
    else:
        result = 'Sehat'
    st.write(result)
    
