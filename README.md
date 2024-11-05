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
|__Jumlah Data__| 15000  |
|__Jumlah variabel__| 22  |

| Nama Data  | Deskripsi data| Encoded data  |
|---|---|---|
|gender| Mengidentifikasi jenis kelamin individu, biasanya dicatat sebagai "Male" atau "Female." | 1 & 0|
|age| variable umur dalam data| - |
|Hypertension| Mengindikasikan apakah individu memiliki riwayat hipertensi (tekanan darah tinggi) | 0 & 1 |
|heart_disease| Mengindikasian apakah individu memiliki riwayat penyakit jantung |0 & 1|
|ever_married| Menikah atau Belum Menikah | - |
|Alcohol Intake| Kebiasaan Mengkonsumsi Alkohol|3, 1, 2, 0|
|work_type| Jenis Pekerjaan |2, 0, 1|
|Residence_type| Sektor Perumahan |0, 1|
|avg_glucose_level| Rata - rata nilai Glucosa individu | - |
|bmi| Berat Badan | - |
|smoking_status| Status pengguna rokok |2, 0, 1|
|Diagnosis| Riwayat Stroke |1, 0|

**Source Dataset :** https://github.com/incribo-inc/stroke_prediction

## Exploratory Data Anlysis (EDA)
Dalam proses Exploratory Data Anlysis (EDA) bertujuan untuk memahami dan menemukan pola dalam data yang digunakan dalam menjelaskan corelasi antar data. Dalam Exploratory Data Analysis dapat dibagi menjadi dua bagian yaitu Univariate dan Multivariate Analysis. 

## Cleaning Data
Cleaning data adalah proses penting dalam analisis data dan machine learning. Tujuan utamanya adalah untuk meningkatkan kualitas data sehingga model yang dibangun dapat menghasilkan prediksi yang lebih akurat dan dapat diandalkan. Dalam tahapan ini ada beberapa cara untuk cleaning data seperti cleaning `missing value`, menghapus data yang memiliki `duplikat`, dan menghapus data yang tidak relevan agar model memiliki prediksi yang akurat. salah satunya adalah terdapat beberapa data yang tidak releven seperti `Never Work` dalam kategori data `Work Type`.

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
Pada Grafik Korelasi diatas dapat korelasi fitur `Diagnosis` dan `Deitary Habits` Memiliki Korelasi yang tinggi, sehingga dapat digunakan untuk melakukan Prediksi Machine Learning.

## Outlier
![New data 4](https://github.com/user-attachments/assets/66b9f86d-c7fb-4c65-a5ac-7292cca042d3)

Gambar .4 Ouliers
Dalam gambar diatas terlihat bahwa data tidak memiliki outliers yang terdeteksi.

## Rubrik / Kriteria Tambahan
1. Memeriksa Struktur dan Ringkasan Data
   Tujuan: Mengetahui ukuran data, tipe data, dan distribusi awal.
   Metode: Gunakan metode seperti .info(), .describe(), .head(). dan lain - lain.
2. Menangani Missing Values
   Tujuan: Mengidentifikasi kolom yang memiliki data hilang (missing) dan mempertimbangkan bagaimana menanganinya.
   Metode: Gunakan .isnull().sum() untuk melihat jumlah nilai yang hilang di setiap kolom.
3. Visualisasi Distribusi Variabel Numerik
   Tujuan: Memahami distribusi setiap variabel numerik dan mendeteksi outlier atau distribusi yang tidak normal.
   Metode: Gunakan histogram atau boxplot untuk setiap variabel numerik.
4. Visualisasi Variabel Kategorikal
   Tujuan: Memahami distribusi dari variabel kategorikal.
   Metode: Gunakan countplot dari seaborn untuk setiap variabel kategorikal.
5. Analisis Korelasi
   Tujuan: Melihat hubungan antar variabel numerik, terutama untuk melihat apakah ada kolinearitas.
   Metode: Gunakan matriks korelasi dan heatmap.

# Data Preperation
## Encoding 
Data diubah ke bentuk Numeric agar dapat digunakan untuk memprediksi Stroke. Mengubah data dapat dilakukan Dengan menggunakan LabelEncoder dari library sklearn atau Dummies pandas yang berupa categoriacal menjadi Numerical.

Dengan Menggunakan LabelEncoder() :
```Python
gender_encoder = LabelEncoder()
diagnosis_encoder = LabelEncoder()
work_type_encoder = LabelEncoder()
smoking_status_encoder = LabelEncoder()
alcohol_intake_encoder = LabelEncoder()
physical_activity_encoder = LabelEncoder()
family_history_encoder = LabelEncoder()
dietary_habits_encoder = LabelEncoder()
residence_type_encoder = LabelEncoder()
blood_pressure_encoder = LabelEncoder()
cholesterol_levels_encoder = LabelEncoder()
```

Penggunaan satu objek LabelEncoder untuk beberapa kolom dapat menyebabkan masalah karena setiap kali kita memanggil fit_transform(), LabelEncoder akan "belajar" label baru dan bisa merusak hasil encoding sebelumnya. Solusi terbaik adalah membuat instance LabelEncoder yang berbeda untuk setiap kolom atau menggunakan teknik mapping manual jika kita ingin memastikan konsistensi label.

```Python
df['Gender'] = gender_encoder.fit_transform(df['Gender'])
df['Diagnosis'] = diagnosis_encoder.fit_transform(df['Diagnosis'])
df['Work Type'] = work_type_encoder.fit_transform(df['Residence Type'])
df['Smoking Status'] = smoking_status_encoder.fit_transform(df['Smoking Status'])
df['Alcohol Intake'] = alcohol_intake_encoder.fit_transform(df['Alcohol Intake'])
df['Physical Activity'] = physical_activity_encoder.fit_transform(df['Physical Activity'])
df['Family History of Stroke'] = family_history_encoder.fit_transform(df['Family History of Stroke'])
df['Dietary Habits'] = dietary_habits_encoder.fit_transform(df['Diagnosis'])
df['Residence Type'] = residence_type_encoder.fit_transform(df['Residence Type'])
df['Blood Pressure Levels'] = blood_pressure_encoder.fit_transform(df['Blood Pressure Levels'])
df['Cholesterol Levels'] = cholesterol_levels_encoder.fit_transform(df['Cholesterol Levels'])
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

Tahapan data preparation sangat penting dalam data analysis dan machine learning, karena data yang tidak bersih atau tidak siap dapat mengarah pada hasil analisis yang salah atau model yang kurang akurat. dalam tahapan ini juga dilakukan proses Encoder, dimana data diubah ke bentuk `Numerical` yang asalnya `Categorical` dengan mengunakan `Encoder`

## Split Dataset
Pada tahap ini, dataset dibagi menjadi dua bagian: data training dan data testing. Data training berfungsi untuk melatih model machine learning. Dalam hal ini, data training dibagi lagi menjadi dua bagian, yaitu data tanpa fitur target `x_train` dan data yang hanya memiliki fitur target `y_train`. Begitu juga dengan data testing, yang dibagi menjadi data tanpa fitur target `x_test` dan data dengan fitur target saja `y_test`. Salah satu metode yang digunakan untuk membagi dataset menjadi empat bagian ini adalah `train_test_split()` dari library `sklearn`. Langkah ini penting untuk menyiapkan data yang diperlukan untuk mengevaluasi model, sehingga pengembang dapat mengetahui akurasi prediksi yang dihasilkan oleh model tersebut. 
Dalam Tahapan ini dilakukan undersampling pada `X` dan `y` terlebih dahulu dengan `RandomUnderSampler()` untuk melakukan balance data, namun kemudian menggunakan `X` dan `y` asli untuk membagi data ke dalam train dan test set menjadi `X_resampled` dan `y_resampled`.

```Python
# split data test dan train
X = df.drop(['Diagnosis'], axis=1)
y = df['Diagnosis']

#  balancing Data
X_resampled, y_resampled = RandomUnderSampler().fit_resample(X,y)

# Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)
```
# Modelling
## Model Random Forest
Pada tahap pemodelan, beberapa algoritma digunakan, salah satunya adalah Random Forest untuk membangun model prediksi. Random Forest adalah algoritma ensemble berbasis pohon keputusan yang bekerja dengan cara membuat sejumlah besar pohon keputusan (decision trees) selama proses pelatihan. Setiap pohon dalam hutan acak dilatih dengan subset acak dari data, baik dalam hal fitur maupun sampel, dan hasil akhirnya didapatkan dengan mengambil rata-rata (untuk regresi) atau melakukan voting (untuk klasifikasi) dari semua pohon tersebut. 
**Parameter Default :**
- max_depth `default = None`: Batas maksimum kedalaman pohon. Jika None, pohon akan terus berkembang sampai semua data terklasifikasi.
- min_samples_split `default = 2`: Jumlah minimum sampel yang diperlukan untuk membagi node.
- criterion `default = 'gini'`: Kriteria yang digunakan untuk menentukan pembagian terbaik (misalnya, 'gini' atau 'entropy').
- min_samples_leaf `default = 1`: Jumlah minimum sampel yang harus ada pada sebuah daun (leaf node).

## Decision Tree
Model ini mudah diinterpretasi dan cepat untuk di-train serta mampu menangani baik data kategorikal maupun numerik, namun cenderung mudah overfitting dan kurang stabil terhadap perubahan kecil pada data.
**Parameter Default :**
- n_neighbors `default = 5`: Jumlah tetangga terdekat yang dipertimbangkan untuk klasifikasi.
- metric `default = 'minkowski'`: Jarak yang digunakan untuk menentukan kedekatan (misalnya, Euclidean atau Manhattan).
- weights `default = 'uniform'`: Menentukan apakah semua tetangga memiliki bobot yang sama `'uniform'` atau diberi bobot berdasarkan jaraknya `'distance'`.

## Adaboost
Algoritma ini meningkatkan akurasi secara adaptif dengan memperhatikan kesalahan dari iterasi sebelumnya sehingga mengurangi bias, tetapi rentan terhadap overfitting jika terdapat outliers atau noise dalam data.
**Parameter Default :**
- n_estimators `default = 50`: Jumlah model lemah `estimators` yang akan digabungkan.
- learning_rate `default = 1.0`: Mengontrol kontribusi masing-masing model terhadap prediksi akhir. Nilai yang lebih rendah membuat model lebih stabil tetapi butuh lebih banyak iterasi.
- base_estimator `default = DecisionTreeClassifier dengan kedalaman maksimum = 1`: Model dasar yang digunakan dalam proses boosting.
 
## KNN(K-Nearest Neighbors)
Algoritma yang sederhana dan non-parametrik ini menghasilkan akurasi tinggi pada data dengan struktur klaster yang baik, tetapi memerlukan waktu komputasi yang lama pada dataset besar dan kurang efektif pada data berdimensi tinggi tanpa feature scaling yang memadai.
**Parameter Default :**
- n_neighbors `default = 5`: Jumlah tetangga terdekat yang dipertimbangkan untuk klasifikasi.
- metric `default = 'minkowski'`: Jarak yang digunakan untuk menentukan kedekatan (misalnya, Euclidean atau Manhattan).
- weights `default = 'uniform'`: Menentukan apakah semua tetangga memiliki bobot yang sama `'uniform'` atau diberi bobot berdasarkan jaraknya `'distance'`.

## Model Hyperparameter Tuning
Hyperparameter Tuning bertujuan untuk meningkatkan kinerja model dengan mengatur parameter-parameter pada algoritma yang digunakan. Dalam kasus ini, pencarian kombinasi hyperparameter terbaik dilakukan menggunakan Grid Search. Metrik yang digunakan untuk menilai performa Grid Search adalah Mean Cross Validation (CV). Setiap kemungkinan kombinasi nilai hyperparameter akan dievaluasi menggunakan metrik CV, dan kombinasi dengan skor CV tertinggi akan diterapkan pada pelatihan model.

## Perbandingan model Setelah menggunakan Hyperparameter Tuning
Jika kita bandingkan dua model K-Nearest Neighbors (KNN) yang berbeda, satu dengan akurasi 0.5 dan satu lagi dengan akurasi 0.9 setelah hyperparameter tuning, kita bisa melihat bahwa tuning parameter memang berpengaruh besar terhadap performa model. 
- Akurasi sebelum dituning mendapat nilai Akurasi 0.4978
- Akurasi Setelah dituning mendapat nilai Akurasi 0.9991


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

## Model Menggunakan Decision Tree
**Akurasi: 1.0**
| Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|-------------|---------|---------|-----------|--------------|
| Precision   | 1.00    | 1.00    | 1.00      | 1.00         |
| Recall      | 1.00    | 1.00    | 1.00      | 1.00         |
| F1-Score    | 1.00    | 1.00    | 1.00      | 1.00         |
| Support     | 1155    | 1131    | 2286      | 2286         |

## Model Menggunakan AdaBoost
**Accuracy**: 1.0
| Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|-------------|---------|---------|-----------|--------------|
| Precision   | 1.00    | 1.00    | 1.00      | 1.00         |
| Recall      | 1.00    | 1.00    | 1.00      | 1.00         |
| F1-Score    | 1.00    | 1.00    | 1.00      | 1.00         |
| Support     | 1155    | 1131    | 2286      | 2286         |

## Model Menggunakan KNN
**Accuracy**: 0.4978
| Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|-------------|---------|---------|-----------|--------------|
| Precision   | 0.50    | 0.49    | 0.50      | 0.50         |
| Recall      | 0.50    | 0.49    | 0.50      | 0.50         |
| F1-Score    | 0.50    | 0.49    | 0.50      | 0.50         |
| Support     | 1155    | 1131    | 2286      | 2286         |

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

## Metrics 
Metrik ini memberikan gambaran yang lebih jelas tentang bagaimana model bekerja dan seberapa baik ia mengklasifikasikan data.

- Akurasi menunjukkan persentase prediksi yang benar dari total prediksi.
- Presisi memberikan informasi tentang seberapa tepat model dalam mengklasifikasikan kelas positif.
- Recall menunjukkan kemampuan model untuk mendeteksi semua instance positif.
- F1-score memberikan gambaran seimbang antara presisi dan recall, yang sangat penting dalam konteks klasifikasi yang mungkin tidak seimbang.

## Korelasi Variabel
Dari grafik matriks korelasi, kita bisa melihat bahwa meskipun tidak semua fitur menunjukkan korelasi yang tinggi dengan diagnosis stroke, hampir semua fitur memiliki korelasi tertentu, meskipun kecil. Misalnya, "Dietary Habits" menunjukkan korelasi yang signifikan dan positif terhadap diagnosis stroke, sehingga menjadi fitur yang paling berpengaruh.

Namun, penting untuk dicatat bahwa meskipun "Dietary Habits" adalah fitur utama, variabel lain seperti "Hypertension," "Heart Disease," dan "Smoking Status" juga memiliki kontribusi yang tidak bisa diabaikan. Kesimpulan ini penting untuk diambil sebagai informasi bahwa faktor-faktor lain turut serta dalam mempengaruhi risiko stroke.

## Evaluasi Business Understanding
- Dengan Univariate dan Multivariate anlysis untuk memahami variable dan hubungan corelasi dengan variable lain, dengan begitu kita dapat menentukan fitur yang akan digunakan dalam model.
- Model yang dibuat dapat melakukan prediksi yang akurat dengan Machine Learning seperti model Decision Tree, Random Forest, Adaboost dan KNN(Hyperparameter Tuning) untuk memprediksi Diagnosis Stroke
- Metrics yang mempengaruhi adalah data yang didapat salah satunya `Dietary Habits`, dan performance model yang digunakan. Sehinggan model terbaik adalah model Decision Tree, Random Forest, Adaboost

**Problem statement** yang diajukan berfokus pada pengembangan model yang mampu memprediksi risiko stroke dengan akurasi tinggi. Berdasarkan hasil evaluasi, model yang diterapkan, terutama Decision Tree, Random Forest, dan AdaBoost, menunjukkan akurasi yang sempurna (1.0). Hal ini membuktikan bahwa model tersebut efektif dalam menjawab pertanyaan awal mengenai kemampuan untuk memprediksi stroke.

**Goals** dari proyek ini adalah untuk meningkatkan kemampuan prediksi dan pemahaman tentang faktor-faktor yang berkontribusi terhadap stroke. Dengan metrik evaluasi seperti presisi, recall, dan F1-score, model yang dikembangkan tidak hanya mampu mengklasifikasikan data dengan akurat tetapi juga memberikan wawasan mengenai variabel yang paling berpengaruh, seperti "Dietary Habits," "Hypertension," "Heart Disease," dan "Smoking Status." Keberhasilan ini menunjukkan bahwa model tidak hanya mencapai, tetapi bahkan melebihi, ekspektasi yang diharapkan.

**Dampak Solusi yang Diterapkan** termasuk penerapan machine learning dengan berbagai algoritma (Decision Tree, Random Forest, AdaBoost, dan KNN dengan hyperparameter tuning), berkontribusi pada pengembangan sistem yang lebih baik untuk memprediksi dan mengelola risiko stroke. Dengan akurasi yang tinggi, sistem ini dapat digunakan oleh praktisi kesehatan untuk memprioritaskan intervensi bagi individu yang berisiko tinggi, mengoptimalkan sumber daya, dan meningkatkan hasil kesehatan.

# Kesimpulan
Kesimpulan dari model Random Forest, Adaboost, Decision Tree, dan KNN(hyperparameter Tuning KNN model) di atas adalah bahwa model ini menunjukkan performa yang sangat baik pada data uji, dengan akurasi, presisi, recall, dan F1-score yang semuanya mencapai nilai maksimum 1.0. Ini menunjukkan bahwa model mampu mengklasifikasikan setiap instance pada data uji dengan benar tanpa ada kesalahan. 

- Membuat model Machine Learning yang dapat melakukan prediksi Diagnosisi Stroke.
- Yang Mempengaruhi Diagnosis Stroke salah satunya adalah Dietary Habits.
- Model dengan Akurasi tinggi adalah model randomforest, adaboost, Decision Tree, dan KNN(Hyperparameter Tuning)

# Referensi
Hasan, T. F., Rabinstein, A. A., Middlebrooks, E. H., Haranhalli, N., Silliman, S. L., Meschia, J. F., & Tawk, R. G. (2018, April). Diagnosis and management of acute ischemic stroke. In Mayo Clinic Proceedings (Vol. 93, No. 4, pp. 523-538). Elsevier.
