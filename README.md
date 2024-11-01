# Laporan Proyek Machine Learning - Juniyara Parisya Setiawan
# Domain Proyek
Stroke adalah penyakit yang terjadi ketika suplai darah ke bagian otak terganggu atau berkurang, menyebabkan jaringan otak kekurangan oksigen dan nutrisi. Beberapa resiko stroke meliputi gaya hidup tidak sehat, seperti merokok, pola makan tinggi lemak, kurang aktivitas fisik, serta tekanan darah tinggi, diabetes, dan kolesterol tinggi. Pola hidup ini sering kali terkait dengan beban kerja yang tinggi, kurangnya waktu untuk berolahraga, dan tingkat stres yang tinggi, yang sering ditemukan pada pekerja di berbagai sektor.

Machine learning dapat menjadi alat yang sangat efektif dalam analisis data kesehatan, karena mampu mengidentifikasi pola kompleks dari data medis yang mungkin sulit diamati oleh manusia. Teknologi ini dapat digunakan untuk memprediksi risiko stroke berdasarkan berbagai faktor, seperti gaya hidup, kondisi medis, dan kebiasaan sehari-hari dari pekerja swasta, PNS, dan pekerja mandiri

# Business Understanding

## Problem Statement
- Variable apa saja yang Berpengaruh dalam diagnosis Stroke ?
- Bagaimana membuat sebuah model untuk memprediksi Seseorang yang memiliki risiko Diagnosis stroke menggunakan data yang didapat ?
-  Metrics apa saja yang mempengaruhi performance model sehingga mendapat model terbaik ?

## Goals
- Mengetahui variable yang memiliki korelasi dengan Stroke.
- Membuat dan membangun model machine learning sederhana untuk memprediksi Diagnosis Stroke dan membuat model terbaik.
- Dapat menentukan Metrics yang mempengaruhi performance model dan mendapat model terbaik.

## Solution Statements
- Dengan Univariate dan Multivariate anlysis untuk memahami variable dan hubungan corelasi dengan variable lain.
- Membangun beberapa model Machine Learning seperti Decision Tree, Random Forest, Adaboost dan lain - lain untuk memprediksi Diagnosis Stroke.
- Menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score untuk menilai performa masing-masing model. (Optional) Hyperparameter tuning dapat dilakukan pada model yang menunjukkan potensi akurasi tinggi untuk lebih meningkatkan performanya jika dibutuhkan.

# Data Understanding
|   |   |
|---|---|
|__Nama dataset__| Medical Stroke Prediction |
|__Deskripsi dataset__| Dataset containing Stroke Prediction metrics |
|__Jumlah sampel__| 15000  |
|__Jumlah variabel__| 22  |

|   |   |
|---|---|
|gender| Mengidentifikasi jenis kelamin individu, biasanya dicatat sebagai "Male" atau "Female." |
|age|
|Hypertension| Mengindikasikan apakah individu memiliki riwayat hipertensi (tekanan darah tinggi) |
|heart_disease| Mengindikasian apakah individu memiliki riwayat penyakit jantung |
|ever_married| Menikah atau Belum Menikah |
|work_type| Jenis Pekerjaan |
|Residence_type| Sektor Perumahan |
|avg_glucose_level| Rata - rata nilai Glucosa individu |
|bmi| Berat Badan |
|smoking_status| Status pengguna rokok |
|stroke| Riwayat Stroke |


# Exploratory Data Anlysis (EDA)
Dalam proses Exploratory Data Anlysis (EDA) bertujuan untuk memahami dan menemukan pola dalam data yang digunakan dalam menjelaskan corelasi antar data. Dalam Exploratory Data Analysis dapat dibagi menjadi dua bagian yaitu Univariate dan Multivariate Analysis.

## Univariate Analysis
![New data 2](https://github.com/user-attachments/assets/22fd2858-c7ae-4df8-99e2-847e918f3cc6)
Gambar .1 Grafik Distribusi
Dalam grafik diatas merupakan distribusi umur. kita dapat melihat bahwa umur 20+ tahun keatas memiliki Stroke.

## Multivariate Analysis
![New data 3](https://github.com/user-attachments/assets/d5050022-e45d-46d5-9bb7-d1a013d238ce)
Gambar .2 Grafik Fitur Gender
Pada grafik tersebut dapat terlihat bahwa Gender Male Memiliki Stroke Paling Banyak.

![New data 1](https://github.com/user-attachments/assets/bea9d180-264e-4bc0-9da4-96fbd819951d)
Gambar .3 Grafik Matriks Korelasi
Pada Grafik Korelasi diatas dapat korelasi fitur 'Diagnosis' dan 'Deitary Habits' Memiliki Korelasi yang tinggi, sehingga dapat digunakan untuk melakukan Prediksi Machine Learning.

# Data Preperation
## Encoding Features
Data diubah ke bentuk Numeric agar dapat digunakan untuk memprediksi Stroke. Mengubah data dapat dilakukan Dengan menggunakan LabelEncoder dari library sklearn atau Dummies pandas yang berupa categoriacal menjadi Numerical.

Dengan Menggunakan LabelEncoder() :
```Python
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Diagnosis'] = encoder.fit_transform(df['Diagnosis'])
df['Work Type'] = encoder.fit_transform(df['Residence Type'])
df['Smoking Status'] = encoder.fit_transform(df['Smoking Status'])
df['Alcohol Intake'] = encoder.fit_transform(df['Alcohol Intake'])
df['Physical Activity'] = encoder.fit_transform(df['Physical Activity'])
df['Family History of Stroke'] = encoder.fit_transform(df['Family History of Stroke'])
df['Dietary Habits'] = encoder.fit_transform(df['Diagnosis'])
df['Residence Type'] = encoder.fit_transform(df['Residence Type'])
df['Blood Pressure Levels'] = encoder.fit_transform(df['Blood Pressure Levels'])
df['Cholesterol Levels'] = encoder.fit_transform(df['Cholesterol Levels'])
df.head()
```
Dengan Menggunakan Pandas.dummies :
```Python
df_encoder = pd.get_dummies(df[['Work Type', 'Residence Type', 'Smoking Status', 'Alcohol Intake', 'Physical Activity', 'Family History of Stroke', 'Dietary Habits', 'Diagnosis']], drop_first=True).astype(int)
df_encoder.head()
```

| Age | Gender | Hypertension | Heart Disease | Work Type | Residence Type | Average Glucose Level | Body Mass Index (BMI) | Smoking Status | Alcohol Intake | Physical Activity | Stroke History | Family History of Stroke | Dietary Habits | Stress Levels | Blood Pressure Levels | Cholesterol Levels | Diagnosis |
|-----|--------|--------------|---------------|-----------|----------------|------------------------|-----------------------|----------------|----------------|-------------------|----------------|--------------------------|----------------|---------------|-----------------------|--------------------|-----------|
| 56  | 1      | 0            | 1             | 0         | 0              | 130.91                 | 22.37                 | 2              | 3              | 2                 | 0              | 1                        | 1              | 3.48          | 1869                  | 4096               | 1         |
| 80  | 1      | 0            | 0             | 1         | 1              | 183.73                 | 32.57                 | 2              | 1              | 1                 | 0              | 0                        | 1              | 1.73          | 2179                  | 3602               | 1         |
| 51  | 1      | 1            | 1             | 1         | 1              | 177.34                 | 29.06                 | 0              | 2              | 1                 | 0              | 1                        | 1              | 6.84          | 1019                  | 3824               | 1         |
| 62  | 0      | 0            | 0             | 1         | 1              | 91.60                  | 37.47                 | 0              | 3              | 0                 | 0              | 0                        | 1              | 4.85          | 1501                  | 5435               | 1         |
| 40  | 0      | 1            | 0             | 1         | 1              | 77.83                  | 28.20                 | 0              | 1              | 1                 | 1              | 0                        | 0              | 6.38          | 3642                  | 138                | 0         |
## Split Dataset
Pada tahap ini, dataset dibagi menjadi dua bagian: data training dan data testing. Data training berfungsi untuk melatih model machine learning. Dalam hal ini, data training dibagi lagi menjadi dua bagian, yaitu data tanpa fitur target `(x_train)` dan data yang hanya memiliki fitur target `(y_train)`. Begitu juga dengan data testing, yang dibagi menjadi data tanpa fitur target `(x_test)` dan data dengan fitur target saja `(y_test)`. Salah satu metode yang digunakan untuk membagi dataset menjadi empat bagian ini adalah `train_test_split()` dari library `sklearn`. Langkah ini penting untuk menyiapkan data yang diperlukan untuk mengevaluasi model, sehingga pengembang dapat mengetahui akurasi prediksi yang dihasilkan oleh model tersebut.
```Python
# split data test dan train
X = df.drop(['Diagnosis'], axis=1)
y = df['Diagnosis']

#  balancing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)
```
# Modelling
## Model Random Forest
Pada tahap pemodelan, beberapa algoritma digunakan, salah satunya adalah Random Forest untuk membangun model prediksi. Random Forest adalah algoritma ensemble berbasis pohon keputusan yang bekerja dengan cara membuat sejumlah besar pohon keputusan (decision trees) selama proses pelatihan. Setiap pohon dalam hutan acak dilatih dengan subset acak dari data, baik dalam hal fitur maupun sampel, dan hasil akhirnya didapatkan dengan mengambil rata-rata (untuk regresi) atau melakukan voting (untuk klasifikasi) dari semua pohon tersebut.

## Model Hyperparameter Tuning
Hyperparameter Tuning bertujuan untuk meningkatkan kinerja model dengan mengatur parameter-parameter pada algoritma yang digunakan. Dalam kasus ini, pencarian kombinasi hyperparameter terbaik dilakukan menggunakan Grid Search. Metrik yang digunakan untuk menilai performa Grid Search adalah Mean Cross Validation (CV). Setiap kemungkinan kombinasi nilai hyperparameter akan dievaluasi menggunakan metrik CV, dan kombinasi dengan skor CV tertinggi akan diterapkan pada pelatihan model.

# Evaluation
Setelah melakukan beberapa pelatihan model `Random Forest` dan model lainnya pada dataset `train_set`, evaluasi dilakukan menggunakan metrik `classification_report` dari library `Scikit-learn` untuk menilai performa model pada dataset test. Fungsi `classification_report` menampilkan beberapa metrik penting, yaitu:

- Presisi digunakan untuk mengukur seberapa dapat diandalkan sebuah model ketika memberikan prediksi terhadap suatu kelas/_target_. 
	Rumus Precison:

$\dfrac{True Positive}{True Positive + False Positive}$
- _Recall_ digunakan untuk mengukur kemampuan model untuk memprediksi kelas _True Positive
	Rumus Recall:

$\dfrac{True Positive}{True Positive + False Negative}$
- _F1-Score_ digunakan untuk mencari titik seimbang antara Presisi dan _Recall
	Rumus F1-Score:

$\dfrac{2 \ast Precision \ast Recall}{Precision+Recall}$

Dengan metrik-metrik ini, performa model dapat dievaluasi secara komprehensif, mengidentifikasi tingkat keakuratan dalam klasifikasi tiap kelas, dan keseimbangan antara presisi serta recall.

## Model Menggunakan Random Forest
**Akurasi: 1.0**

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Class 0      | 1.00      | 1.00   | 1.00     | 1155    |
| Class 1      | 1.00      | 1.00   | 1.00     | 1131    |
| **Accuracy** |           |        | 1.00     | 2286    |
| Macro Avg    | 1.00      | 1.00   | 1.00     | 2286    |
| Weighted Avg | 1.00      | 1.00   | 1.00     | 2286    |

- Akurasi Model: Model ini mencapai akurasi 1.0 atau 100%, yang menunjukkan bahwa semua prediksi model pada data uji adalah benar.

- Presisi: Presisi untuk kedua kelas (0 dan 1) mencapai 1.0, yang berarti model tidak membuat kesalahan dalam memprediksi instance sebagai positif atau negatif untuk masing-masing kelas. Presisi tinggi ini menunjukkan bahwa model mampu membedakan kelas dengan sangat baik.

- Recall: Recall untuk kedua kelas juga mencapai 1.0, yang menunjukkan bahwa model mampu mendeteksi seluruh instance dari masing-masing kelas dengan benar tanpa ada yang terlewatkan. Ini berarti model tidak melewatkan instance positif atau negatif.

- F1-Score: Dengan presisi dan recall yang sama-sama mencapai 1.0, F1-score untuk kedua kelas juga berada pada 1.0, menunjukkan keseimbangan yang sempurna antara presisi dan recall.

- Macro dan Weighted Average: Macro average dan weighted average untuk presisi, recall, dan F1-score semuanya adalah 1.0, menunjukkan bahwa performa model stabil di seluruh kelas tanpa adanya bias terhadap salah satu kelas.

## Model Menggunakan Decision Tree
**Akurasi: 1.0**
| Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|-------------|---------|---------|-----------|--------------|
| Precision   | 1.00    | 1.00    | 1.00      | 1.00         |
| Recall      | 1.00    | 1.00    | 1.00      | 1.00         |
| F1-Score    | 1.00    | 1.00    | 1.00      | 1.00         |
| Support     | 1155    | 1131    | 2286      | 2286         |

- Precision Ini adalah proporsi prediksi benar dari total prediksi positif yang dibuat oleh model. Class 0 dan Class 1 masing-masing memiliki precision 1.00, yang berarti model tidak menghasilkan kesalahan dalam prediksi kelas positifnya.

- Recall Ini menunjukkan seberapa baik model menangkap semua contoh aktual dari setiap kelas. Nilai recall 1.00 untuk kedua kelas menunjukkan bahwa model berhasil mengidentifikasi semua contoh untuk kedua kelas tanpa ada yang terlewat.
  
- F1-Score Ini adalah rata-rata harmonik dari precision dan recall, yang berguna untuk mengukur performa model dalam menangani ketidakseimbangan antar kelas. F1-score 1.00 untuk kedua kelas menunjukkan model memiliki performa sangat baik tanpa trade-off antara precision dan recall.
  
- Support Ini adalah jumlah contoh dalam data untuk masing-masing kelas. Support 1155 untuk Class 0 dan 1131 untuk Class 1 menunjukkan jumlah data masing-masing kelas, dengan total 2286.

- Accuracy: Akurasi model adalah 1.0, yang menunjukkan bahwa model Decision Tree berhasil memprediksi setiap data uji dengan benar tanpa kesalahan.

## Model Menggunakan AdaBoost
**Accuracy**: 1.0
| Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|-------------|---------|---------|-----------|--------------|
| Precision   | 1.00    | 1.00    | 1.00      | 1.00         |
| Recall      | 1.00    | 1.00    | 1.00      | 1.00         |
| F1-Score    | 1.00    | 1.00    | 1.00      | 1.00         |
| Support     | 1155    | 1131    | 2286      | 2286         |

- AdaBoost sering menggunakan decision stumps atau decision trees sederhana sebagai base learner
- dataset memiliki kelas-kelas yang sangat mudah dipisahkan dengan batas yang jelas di antara keduanya, AdaBoost akan lebih mudah mencapai akurasi sempurna
- Karena dataset memiliki relevansi yang tinggi maka model dapat belajar.

## Model Menggunakan KNN
**Accuracy**: 0.4978
| Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|-------------|---------|---------|-----------|--------------|
| Precision   | 0.50    | 0.49    | 0.50      | 0.50         |
| Recall      | 0.50    | 0.49    | 0.50      | 0.50         |
| F1-Score    | 0.50    | 0.49    | 0.50      | 0.50         |
| Support     | 1155    | 1131    | 2286      | 2286         |

- KNN bekerja dengan mencari tetangga terdekat dari setiap data berdasarkan jarak di ruang fitur. Jika data dari kedua kelas (Class 0 dan Class 1) berada sangat dekat atau saling tumpang tindih di ruang fitur, maka model akan kesulitan membedakan kelas tersebut. Hal ini membuat prediksi KNN mendekati tebak-tebakan acak
- Jika K terlalu kecil, model dapat menjadi sangat sensitif terhadap noise. Sehingga mendapat Akurasi yang rendah
- Jika dataset memiliki banyak fitur, efek dari jarak Euclidean menjadi kurang efektif untuk memisahkan kelas

## Model Hyperparameter Tuning
dengan menggunakan Hypermeter Tuning Kita dapat melihat akurasi yang signifikan terhadapat Model. **Nilai terbaik yang diperoleh oleh Hyperparameter Tuning adalah:** 

`Best Hyperparameters: {'algorithm': 'brute', 'n_neighbors': 11, 'weights': 'uniform'}`
`Best Cross-Validation Score: 1.0`

**Accuracy**: 0.9991

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| 0     | 1.00      | 1.00   | 1.00     | 1155   |
| 1     | 1.00      | 1.00   | 1.00     | 1131   |
| **Accuracy**       | -         | -      | **1.00**     | **2286**   |
| **Macro Avg**     | 1.00      | 1.00   | 1.00     | 2286   |
| **Weighted Avg**  | 1.00      | 1.00   | 1.00     | 2286   |

Dalam program di atas, hyperparameter tuning dilakukan pada model K-Nearest Neighbors (KNN) menggunakan GridSearchCV dari pustaka sklearn
- Parameter ini menentukan jumlah tetangga terdekat yang akan dipertimbangkan untuk menentukan kelas dari titik data yang baru.
- Dalam grid search ini, nilai yang diuji adalah: [3, 5, 7, 9, 11, 15, 20]. Memilih jumlah tetangga yang tepat penting untuk mencapai keseimbangan antara bias dan varians.
- GridSearchCV Digunakan untuk menguji kombinasi dari berbagai nilai hyperparameter di atas. Dengan menggunakan metode cross-validation `(cv=5)`, model dievaluasi menggunakan 5 bagian data yang berbeda untuk memastikan bahwa hasil yang diperoleh tidak bergantung pada pembagian data tertentu.
- Scoring dilakukan berdasarkan scoring='accuracy', yang berarti model yang memiliki akurasi tertinggi pada data pelatihan (setelah penyeimbangan kelas dengan SMOTE) akan dipilih sebagai model terbaik

# Kesimpulan
Kesimpulan dari model Random Forest, Adaboost, Decision Tree, dan KNN(hyperparameter Tuning KNN model) di atas adalah bahwa model ini menunjukkan performa yang sangat baik pada data uji, dengan akurasi, presisi, recall, dan F1-score yang semuanya mencapai nilai maksimum 1.0. Ini menunjukkan bahwa model mampu mengklasifikasikan setiap instance pada data uji dengan benar tanpa ada kesalahan.

- Membuat model Machine Learning yang dapat melakukan prediksi Diagnosisi Stroke.
- Yang Mempengaruhi Diagnosis Stroke salah satunya adalah Dietary Habits.
- Model dengan Akurasi tinggi adalah model randomforest, adaboost, Decision Tree, dan KNN(Hyperparameter Tuning)

## Perbandingan model Setelah menggunakan Hyperparameter Tuning
Jika kita bandingkan dua model K-Nearest Neighbors (KNN) yang berbeda, satu dengan akurasi 0.5 dan satu lagi dengan akurasi 0.9 setelah hyperparameter tuning, kita bisa melihat bahwa tuning parameter memang berpengaruh besar terhadap performa model. 
- Akurasi sebelum dituning mendapat nilai Akurasi 0.4978
- Akurasi Setelah dituning mendapat nilai Akurasi 0.9991

## Evaluasi Business Understanding
- Dengan Univariate dan Multivariate anlysis untuk memahami variable dan hubungan corelasi dengan variable lain, dengan begitu kita dapat menentukan fitur yang akan digunakan dalam model.
- Model yang dibuat dapat melakukan prediksi yang akurat dengan Machine Learning seperti model Decision Tree, Random Forest, Adaboost dan KNN(Hyperparameter Tuning) untuk memprediksi Diagnosis Stroke
- Metrics yang mempengaruhi adalah data yang didapat salah satunya `Dietary Habits`, dan performance model yang digunakan. Sehinggan model terbaik adalah model Decision Tree, Random Forest, Adaboost



# Referensi
Hasan, T. F., Rabinstein, A. A., Middlebrooks, E. H., Haranhalli, N., Silliman, S. L., Meschia, J. F., & Tawk, R. G. (2018, April). Diagnosis and management of acute ischemic stroke. In Mayo Clinic Proceedings (Vol. 93, No. 4, pp. 523-538). Elsevier.
