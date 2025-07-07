import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import io
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Analisis Faktor Kemiskinan Indonesia", layout="wide")
st.title("Aplikasi Analisis Faktor-Faktor yang Mempengaruhi Tingkat Kemiskinan di Indonesia")
st.write("Metode: **Elastic Net Regression**")

menu = st.sidebar.radio(
    "Pilih Menu",
    ("Upload Data", "EDA", "Preprocessing")
)

# Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload file Excel berisi data Anda", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("Data berhasil diunggah!")

    if menu == "Upload Data":
        st.header("📂 Upload Data")
        st.dataframe(df)

    elif menu == "EDA":
        st.header("📊 Exploratory Data Analysis (EDA)")

        st.subheader("Informasi Data")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

        st.subheader("Deskriptif Statistik")
        st.write(df.describe())

        st.subheader("Heatmap Korelasi")
        corr_matrix = df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Distribusi Variabel Numerik")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Distribusi {col}')
            st.pyplot(fig)

        if 'Persentase Penduduk Miskin' in df.columns:
            st.subheader("Scatter Plot: Variabel vs Persentase Penduduk Miskin")
            y_col = 'Persentase Penduduk Miskin'
            indep_cols = [col for col in numeric_cols if col != y_col]
            for col in indep_cols:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.scatterplot(x=df[col], y=df[y_col], ax=ax)
                ax.set_title(f'{col} vs {y_col}')
                st.pyplot(fig)

        st.subheader("Pairplot Multivariat")
        fig = sns.pairplot(df.select_dtypes(include=np.number), diag_kind='kde', corner=True)
        st.pyplot(fig)

        st.subheader("Clustered Heatmap Korelasi")
        fig = sns.clustermap(df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
        st.pyplot(fig.fig)

    elif menu == "Preprocessing":
        st.header("⚙️ Preprocessing Data")

        st.subheader("Identifikasi Missing Value")
        missing = df.isnull().sum()
        st.write(missing)

        st.subheader("Identifikasi Outlier (IQR)")
        def identify_outliers_iqr(df_):
            outlier_counts = {}
            for col in df_.select_dtypes(include=np.number).columns:
                Q1 = df_[col].quantile(0.25)
                Q3 = df_[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = df_[(df_[col] < lower) | (df_[col] > upper)]
                outlier_counts[col] = outliers.shape[0]
            return pd.Series(outlier_counts)

        outlier_summary = identify_outliers_iqr(df)
        st.write("Jumlah Outlier per Kolom:")
        st.write(outlier_summary)

        st.subheader("Boxplot Sebelum Winsorizing")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df.select_dtypes(include=np.number), ax=ax)
        ax.set_title('Boxplot Sebelum Winsorizing')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Proses Winsorizing Manual (2.5% ekor)")
        df_winsorized = df.copy()
        for col in df_winsorized.select_dtypes(include=np.number).columns:
            lower_p = np.percentile(df_winsorized[col], 2.5)
            upper_p = np.percentile(df_winsorized[col], 97.5)
            df_winsorized[col] = np.where(df_winsorized[col] < lower_p, lower_p, df_winsorized[col])
            df_winsorized[col] = np.where(df_winsorized[col] > upper_p, upper_p, df_winsorized[col])

        st.write("Data Setelah Winsorizing (5 baris pertama):")
        st.write(df_winsorized.head())

        st.subheader("Boxplot Setelah Winsorizing")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_winsorized.select_dtypes(include=np.number), ax=ax)
        ax.set_title('Boxplot Setelah Winsorizing')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.success("Preprocessing Data Selesai! Anda siap melanjutkan ke modeling Elastic Net Regression.")

else:
    st.info("Silakan upload dataset Anda di sidebar untuk memulai analisis.")
