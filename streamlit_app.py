import optuna
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import io
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, ElasticNet
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')


st.set_page_config(page_title="Analisis Kemiskinan dengan Regularisasi", layout="wide")
st.title("Aplikasi Analisis Tingkat Kemiskinan di Indonesia")

# Menu navigasi aplikasi + logo di sidebar
st.sidebar.image("LOGO.png", width=100)  # Logo tampil di sidebar, ukuran kecil (100px)
st.sidebar.header("Menu")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Beranda", "Upload Data", "EDA", "Preprocessing", "Pemodelan"]
)

if menu == "Beranda":
    # 📝 Pengantar aplikasi hanya muncul di halaman Beranda
    st.markdown("""
    📊 **Selamat Datang di Aplikasi Analisis Tingkat Kemiskinan di Indonesia**

    Aplikasi ini merupakan alat bantu interaktif berbasis data untuk menganalisis faktor-faktor yang memengaruhi persentase penduduk miskin di berbagai daerah di Indonesia.  
    Melalui proses pemodelan regresi seperti Linear, Ridge, Lasso, Elastic Net, hingga Elastic Net yang dioptimasi dengan Optuna, aplikasi ini membantu pengguna memahami hubungan antara variabel-variabel sosial-ekonomi dengan tingkat kemiskinan.

    🎯 **Tujuan Penggunaan Aplikasi:**  
    Aplikasi ini bertujuan untuk memudahkan analisis data kemiskinan dengan antarmuka yang sederhana, sehingga pengguna dapat mengidentifikasi variabel yang paling signifikan memengaruhi tingkat kemiskinan dan mendukung pengambilan keputusan berbasis data dalam upaya penanggulangan kemiskinan di Indonesia.

    📌 **Langkah Awal:**  
    Silakan unggah dataset Anda terlebih dahulu untuk memulai analisis.
    """)
    
# UPLOAD DATA
if menu == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Upload file Excel berisi data Anda", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df  # Simpan di session state agar menu lain bisa pakai
        st.success("Data berhasil diunggah!")
        st.dataframe(df)
    else:
        st.info("Silakan upload dataset Anda di sini.")

# EDA
elif menu == "EDA":
    if "df" in st.session_state:
        df = st.session_state.df

        # Info Data
        st.subheader("Informasi Data")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

        # Deskriptif Statistik
        st.subheader("Deskriptif Statistik")
        st.write(df.describe())

        # Analisis Korelasi
        st.subheader("Heatmap Korelasi")
        corr_matrix = df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)
        st.pyplot(fig)

        # Distribusi Variabel
        st.subheader("Distribusi Variabel Numerik")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Distribusi {col}')
            st.pyplot(fig)

        # Scatter Plot
        if 'Persentase Penduduk Miskin' in df.columns:
            st.subheader("Scatter Plot: Variabel vs Persentase Penduduk Miskin")
            y_col = 'Persentase Penduduk Miskin'
            indep_cols = [col for col in numeric_cols if col != y_col]
            for col in indep_cols:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.scatterplot(x=df[col], y=df[y_col], ax=ax)
                ax.set_title(f'{col} vs {y_col}')
                st.pyplot(fig)

        # Pairplot
        st.subheader("Pairplot Multivariat")
        fig = sns.pairplot(df.select_dtypes(include=np.number), diag_kind='kde', corner=True)
        st.pyplot(fig)

        # Clustered Heatmap
        st.subheader("Clustered Heatmap Korelasi")
        fig = sns.clustermap(df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
        st.pyplot(fig.fig)
    else:
        st.warning("Silakan upload data terlebih dahulu di menu Upload Data.")

# PREPROCESSING
elif menu == "Preprocessing":
    if "df" in st.session_state:
        df = st.session_state.df.copy()

        st.header("Identifikasi Missing Value")
        st.write(df.isnull().sum())

        st.header("Identifikasi Outlier dengan IQR")
        def identify_outliers_iqr(df):
            outlier_counts = {}
            for col in df.select_dtypes(include=np.number).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_counts[col] = outliers.shape[0]
            return pd.Series(outlier_counts)

        outliers_count = identify_outliers_iqr(df)
        st.write("Jumlah Outlier per Kolom (IQR > 1.5):")
        st.write(outliers_count)

        st.subheader("Boxplot Sebelum Winsorizing")
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(data=df.select_dtypes(include=np.number), ax=ax)
        ax.set_title('Boxplot Sebelum Winsorizing')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        st.header("Winsorizing Manual (2.5% di kedua ekor)")
        df_winsor = df.copy()
        for col in df_winsor.select_dtypes(include=np.number).columns:
            lower, upper = np.percentile(df_winsor[col], 2.5), np.percentile(df_winsor[col], 97.5)
            df_winsor[col] = np.clip(df_winsor[col], lower, upper)
        st.session_state.df_winsor = df_winsor
        st.write("Data setelah winsorizing (5 baris pertama):")
        st.dataframe(df_winsor.head())

        st.subheader("Boxplot Setelah Winsorizing")
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(data=df_winsor.select_dtypes(include=np.number), ax=ax)
        ax.set_title('Boxplot Setelah Winsorizing')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        # Update df_analysis dengan data yang sudah di-winsorize
        df_analysis = df_winsor.copy()

        X = df_analysis.drop('Persentase Penduduk Miskin', axis=1)
        y = df_analysis['Persentase Penduduk Miskin']

        # PASTIKAN hanya kolom numerik untuk VIF
        X_num = X.select_dtypes(include=[np.number])

        st.header("Uji Multikolinearitas (VIF)")

        def calculate_vif(X_input):
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_input.columns
            vif_data["VIF"] = [variance_inflation_factor(X_input.values, i) for i in range(X_input.shape[1])]
            return vif_data

        try:
            vif = calculate_vif(X_num)
            st.write(vif)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='feature', y='VIF', data=vif, ax=ax)
            ax.set_title('Variance Inflation Factor (VIF)')
            ax.axhline(y=10, color='r', linestyle='--', label='Threshold (VIF=10)')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi error saat menghitung VIF: {e}")

    else:
        st.error("Silakan upload data terlebih dahulu di menu Upload Data.")

# PEMODELAN
elif menu == "Pemodelan":
    if "df_winsor" in st.session_state:
        df = st.session_state.df_winsor.copy()
        st.header("Pengaturan Pemodelan")

        # Pilih rasio splitting
        split_ratio = st.selectbox("Pilih rasio split data train:test", ["70:30", "80:20", "90:10"])
        test_size = {"70:30": 0.3, "80:20": 0.2, "90:10": 0.1}[split_ratio]

        # Pilih model
        model_choice = st.selectbox("Pilih Model", ["Linear", "Ridge", "Lasso", "Elastic Net", "Elastic Net Optuna"])

        # Pilih parameter
        if model_choice in ["Ridge", "Lasso", "Elastic Net"]:
            st.markdown("""
            **Apa itu Alpha (λ)?**

            Alpha adalah parameter regularisasi yang mengontrol kekuatan penalti pada koefisien regresi.
            - Semakin besar alpha ➔ penalti regularisasi semakin kuat ➔ koefisien cenderung semakin mendekati nol ➔ model lebih sederhana ➔ risiko underfitting meningkat.
            - Semakin kecil alpha ➔ penalti lebih lemah ➔ koefisien lebih bebas ➔ risiko overfitting meningkat.
            """)
            alpha = st.number_input("Alpha (λ) untuk regularisasi", min_value=0.00001, max_value=10.0, value=1.0, format="%.5f")

            if model_choice == "Elastic Net":
                st.markdown("""
                **Apa itu l1_ratio pada Elastic Net?**

                l1_ratio mengatur keseimbangan antara penalti L1 (Lasso) dan L2 (Ridge):
                - l1_ratio = 1 ➔ seperti Lasso ➔ lebih banyak koefisien bisa jadi nol (fitur dieliminasi).
                - l1_ratio = 0 ➔ seperti Ridge ➔ semua koefisien diperkecil tapi tidak menjadi nol.
                - Semakin mendekati 1 ➔ efek L1 lebih dominan ➔ lebih agresif dalam seleksi variabel.
                """)
                l1_ratio = st.slider("l1_ratio Elastic Net", 0.01, 1.0, 0.5)
       
        elif model_choice == "Elastic Net Optuna":
            st.markdown("""
            **Apa itu Jumlah Trial Optuna?**

            Jumlah trial menentukan berapa kali Optuna mencoba kombinasi parameter yang berbeda untuk menemukan pengaturan terbaik (best hyperparameters).
            - Semakin besar jumlah trial ➔ Optuna mengeksplorasi lebih banyak kombinasi parameter ➔ peluang menemukan parameter optimal semakin tinggi.
            - Tetapi jumlah trial yang besar ➔ waktu komputasi juga semakin lama.
            """)
            n_trials = st.number_input("Jumlah trial Optuna", min_value=10, max_value=5000, value=100, step=10)

        X = df.drop('Persentase Penduduk Miskin', axis=1)
        y = df['Persentase Penduduk Miskin']

        # PILIH HANYA KOLON NUMERIK UNTUK MODELING
        X_num = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if st.button("Jalankan Pemodelan"):
            if model_choice == "Linear":
                model = LinearRegression().fit(X_train_scaled, y_train)
            elif model_choice == "Ridge":
                model = RidgeCV(alphas=[alpha]).fit(X_train_scaled, y_train)
            elif model_choice == "Lasso":
                model = LassoCV(alphas=[alpha], max_iter=10000).fit(X_train_scaled, y_train)
            elif model_choice == "Elastic Net":
                model = ElasticNetCV(l1_ratio=[l1_ratio], alphas=[alpha], max_iter=10000).fit(X_train_scaled, y_train)
            elif model_choice == "Elastic Net Optuna":
                def objective(trial):
                    alpha_opt = trial.suggest_float('alpha', 1e-5, 1.0, log=True)
                    l1_ratio_opt = trial.suggest_float('l1_ratio', 0.01, 1.0)
                    model_opt = ElasticNet(alpha=alpha_opt, l1_ratio=l1_ratio_opt, max_iter=10000, random_state=42)
                    return np.mean(cross_val_score(model_opt, X_train_scaled, y_train, scoring="r2", cv=5))

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials)
                best_alpha, best_l1_ratio = study.best_params['alpha'], study.best_params['l1_ratio']
                st.write("**Parameter Terbaik dari Optuna (Elastic Net):**")
                st.write(f"- alpha: {best_alpha:.6f}")
                st.write(f"- l1_ratio: {best_l1_ratio:.6f}")
                st.write(f"**Skor terbaik dari optimasi:** {study.best_value:.4f}")
                model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000, random_state=42).fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            st.subheader("Hasil Evaluasi Model")
            st.write(f"MAE: {metrics.mean_absolute_error(y_test, y_pred):.4f}")
            st.write(f"MSE: {metrics.mean_squared_error(y_test, y_pred):.4f}")
            st.write(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.4f}")
            st.write(f"R²: {metrics.r2_score(y_test, y_pred):.4f}")
            st.write("Intercept:", model.intercept_)
            st.write("Koefisien:", model.coef_)

            # TAMPILKAN NAMA VARIABEL + KOEFISIEN
            coef_df = pd.DataFrame({
                "Feature": X_num.columns,
                "Coefficient": model.coef_
            })

            st.subheader(f"Koefisien Model {model_choice}")
            st.dataframe(coef_df)

            if model_choice == "Lasso":
                eliminated = coef_df[coef_df['Coefficient'] == 0]
                remaining = coef_df[coef_df['Coefficient'] != 0]

                st.subheader("Variabel Dieliminasi oleh Lasso (Koefisien = 0)")
                if eliminated.empty:
                    st.write("Tidak ada variabel yang dieliminasi. Semua variabel tetap digunakan model.")
                else:
                    st.dataframe(eliminated)

                st.subheader("Variabel yang Tetap Digunakan (Koefisien ≠ 0)")
                st.dataframe(remaining)

    else:
        st.error("Silakan lakukan preprocessing terlebih dahulu di menu Preprocessing.")
        st.success("Preprocessing Data Selesai! Anda siap melanjutkan ke modeling Elastic Net Regression.")

else:
    st.info("Silakan upload dataset Anda di sidebar untuk memulai analisis.")


