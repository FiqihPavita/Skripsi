import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import acf, pacf # Meskipun ACF tidak ditampilkan, statsmodels mungkin masih dibutuhkan untuk PACF
from statsmodels.graphics.tsaplots import plot_pacf # Hanya plot_pacf yang digunakan
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
# 0. Helper Functions and Classes (with updated plot colors)
# -----------------------------------------------------------------------------

# --- Definisi Warna Tema Gelap ---
BG_COLOR = "#12403C"
CONTENT_BG_COLOR = "#12403C"
TEXT_COLOR = "#E2E8F0"
ACCENT_COLOR = "#D2B48C"
PLOT_LINE_COLOR = "#F6F4F0"
PLOT_GRID_COLOR = "#4A5568"

class DataPreprocessing:
    def __init__(self, data_column_name='IHK', feature_range=(0, 1)):
        self.data_column_name = data_column_name
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.feature_range = feature_range
        self._data_for_fitting_scaler = None

    def check_missing_values(self, data_df):
        missing = data_df[[self.data_column_name]].isnull().sum()
        total_missing = missing.sum()
        return total_missing, missing[missing > 0]

    def fit_scaler(self, data_df):
        self._data_for_fitting_scaler = data_df[[self.data_column_name]].copy()
        self.scaler.fit(self._data_for_fitting_scaler)

    def normalize(self, data_df_column):
        return self.scaler.transform(data_df_column)

    def denormalize(self, normalized_data):
        return self.scaler.inverse_transform(normalized_data)

def plot_pacf_only_streamlit(series, lags=20, series_name="Series"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), facecolor=CONTENT_BG_COLOR) # Gunakan warna tema
    
    plot_pacf(series, lags=lags, ax=ax, method='ywm', color=PLOT_LINE_COLOR, vlines_kwargs={"colors": PLOT_GRID_COLOR})
    ax.set_title(f'Partial Autocorrelation Function (PACF) - {series_name}', color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.set_facecolor(CONTENT_BG_COLOR) # Latar belakang axis
    for spine in ax.spines.values(): # Warna bingkai
        spine.set_edgecolor(PLOT_GRID_COLOR)
    
    plt.tight_layout()
    return fig

def create_supervised_data(data_series, lags):
    df = pd.DataFrame(data_series)
    df.columns = ['value']
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    df.dropna(inplace=True)
    X = df[[f'lag_{lag}' for lag in lags]].values
    y = df['value'].values
    return X, y, df.index

class RBFNN:
    def __init__(self, n_centers, activation='gaussian'):
        self.n_centers = n_centers
        self.activation = activation
        self.centers = None
        self.weights = None
        self.spread = None

    def _gaussian(self, X_row, c):
        distance = np.linalg.norm(X_row - c)
        if self.spread == 0:
            return 1.0 if distance == 0 else 0.0
        return np.exp(-(distance**2) / (2 * (self.spread**2)))

    def _multiquadric(self, X_row, c):
        return np.sqrt(np.linalg.norm(X_row - c) ** 2 + self.spread ** 2)

    def _calculate_activations(self, X):
        activations_list = []
        for i in range(X.shape[0]):
            row_activations = [self._gaussian(X[i,:], c) if self.activation == 'gaussian' else self._multiquadric(X[i,:], c) for c in self.centers]
            activations_list.append(row_activations)
        
        activations_matrix = np.array(activations_list)
        if activations_matrix.ndim == 1 and X.shape[0] > 0:
             activations_matrix = activations_matrix.reshape(X.shape[0], -1)
        elif X.shape[0] == 0:
            return np.empty((0, self.n_centers))
        return activations_matrix

    def fit(self, X, y):
        if X.shape[0] < self.n_centers:
            st.error(f"Jumlah data ({X.shape[0]}) lebih kecil dari jumlah center ({self.n_centers}).")
            return False

        kmeans = KMeans(n_clusters=self.n_centers, random_state=42, n_init='auto')
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        
        if self.n_centers > 1:
            center_distances = cdist(self.centers, self.centers, metric='euclidean')
            d_max = np.max(center_distances)
            self.spread = d_max / np.sqrt(2 * self.n_centers)
            if self.spread == 0:
                self.spread = np.mean(np.std(X, axis=0)) if X.shape[0] > 0 else 1.0
                if self.spread == 0: self.spread = 1e-6
                st.warning(f"Spread dihitung ulang karena d_max=0. Spread baru: {self.spread:.4f}")
        elif self.n_centers == 1:
            self.spread = np.mean(np.std(X, axis=0)) if X.ndim > 1 else np.std(X)
            if self.spread == 0: self.spread = 1e-6
        else:
            st.error("Jumlah center harus minimal 1.")
            return False

        activations = self._calculate_activations(X)
        activations_with_bias = np.hstack([activations, np.ones((activations.shape[0], 1))])
        y_reshaped = np.array(y).reshape(-1, 1)
        self.weights = np.linalg.pinv(activations_with_bias) @ y_reshaped
        return True

    def predict(self, X):
        if self.centers is None:
            st.error("Model belum dilatih.")
            return np.array([])
        activations = self._calculate_activations(X)
        activations_with_bias = np.hstack([activations, np.ones((activations.shape[0], 1))])
        return activations_with_bias @ self.weights

def smape(y_true, y_pred):
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    epsilon = 1e-9
    return 2 * np.mean(np.abs(y_pred_flat - y_true_flat) / (np.abs(y_pred_flat) + np.abs(y_true_flat) + epsilon)) * 100

def predict_future(model, initial_data_normalized, lags, steps, normalizer):
    predictions_normalized = []
    current_input_normalized = initial_data_normalized.copy().flatten()

    for _ in range(steps):
        input_for_prediction = np.array([current_input_normalized[-lag] for lag in lags]).reshape(1, -1)
        pred_normalized = model.predict(input_for_prediction)
        pred_normalized_scalar = pred_normalized[0,0] if isinstance(pred_normalized, np.ndarray) and pred_normalized.size > 0 else 0.0
        predictions_normalized.append(pred_normalized_scalar)
        current_input_normalized = np.append(current_input_normalized, pred_normalized_scalar)

    denorm_predictions = normalizer.denormalize(np.array(predictions_normalized).reshape(-1, 1))
    return denorm_predictions.flatten()

# -----------------------------------------------------------------------------
# Streamlit App Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="📈 Prediksi IHK RBFNN", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for dark theme
st.markdown(f"""
<style>
    body {{ color: {TEXT_COLOR}; }}
    .stApp {{ background-color: {BG_COLOR}; }}
    .main .block-container {{ background-color: {CONTENT_BG_COLOR}; border-radius: 10px; padding: 2rem; box-shadow: 0 4px 8px rgba(0,0,0,0.3); border: 1px solid {PLOT_GRID_COLOR}; }}
    h1, h2, h3, h4, h5, h6 {{ color: {ACCENT_COLOR}; }}
    .stButton>button {{ background-color: #3182CE; color: white; border-radius: 5px; border: 1px solid #2B6CB0; padding: 0.5rem 1rem; }}
    .stButton>button:hover {{ background-color: #2B6CB0; color: white; }}
    .stButton>button:focus {{ outline: none !important; box-shadow: 0 0 0 0.2rem rgba(49,130,206,.5) !important; }}
    div[data-testid="stHorizontalBlock"] > div[data-testid^="stVerticalBlock"] > div[data-testid^="stButton"] > button {{ width: 100%; background-color: transparent; color: #A0AEC0; border: 1px solid {PLOT_GRID_COLOR}; }}
    div[data-testid="stHorizontalBlock"] > div[data-testid^="stVerticalBlock"] > div[data-testid^="stButton"] > button:hover {{ background-color: {PLOT_GRID_COLOR}; color: #FFFFFF; }}
    [data-testid="stSidebar"] {{ background-color: {CONTENT_BG_COLOR}; padding: 10px; border-right: 1px solid {PLOT_GRID_COLOR}; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {ACCENT_COLOR}; }}
    [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {{ color: {TEXT_COLOR}; }} /* Ganti selector jika perlu */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stRadio div[role="radiogroup"] {{ 
        background-color: {BG_COLOR} !important; 
        color: {TEXT_COLOR} !important; 
        border: 1px solid {PLOT_GRID_COLOR} !important; 
        border-radius: 5px;
    }}
    .stRadio div[role="radiogroup"] {{ /* Khusus untuk st.radio agar paddingnya benar */
        padding: 0.5rem; 
    }}
    .stDataFrame {{ border: 1px solid {PLOT_GRID_COLOR}; border-radius: 5px; }}
    a {{ color: #63B3ED; }}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Beranda"
if 'data_timeseries' not in st.session_state:
    st.session_state.data_timeseries = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'normalizer' not in st.session_state:
    st.session_state.normalizer = DataPreprocessing(data_column_name='IHK')
if 'normalized_series' not in st.session_state:
    st.session_state.normalized_series = None
if 'significant_lags_input' not in st.session_state:
    st.session_state.significant_lags_input = "1, 2, 3"
if 'trained_rbf_model' not in st.session_state:
    st.session_state.trained_rbf_model = None
if 'model_params' not in st.session_state:
    st.session_state.model_params = {}
if 'data_column_name' not in st.session_state:
    st.session_state.data_column_name = 'IHK'
if 'date_column_name' not in st.session_state:
    st.session_state.date_column_name = 'Tanggal'

# Navigation
st.markdown(f"<h1 style='text-align: center; color: {ACCENT_COLOR};'>📈 Aplikasi Prediksi IHK dengan RBFNN</h1>", unsafe_allow_html=True)
st.markdown("---")

nav_cols = st.columns(5)
pages = ["Beranda", "Upload File", "Pre-processing", "Pemodelan", "Prediksi"]
for i, page_name in enumerate(pages):
    disabled_page = (page_name in ["Pre-processing", "Pemodelan", "Prediksi"]) and not st.session_state.file_uploaded
    if nav_cols[i].button(page_name, key=f"nav_{page_name}", disabled=disabled_page, use_container_width=True):
        st.session_state.current_page = page_name
        st.rerun()

if (st.session_state.current_page in ["Pre-processing", "Pemodelan", "Prediksi"]) and not st.session_state.file_uploaded:
    st.warning("⛔ Harap upload file CSV terlebih dahulu pada halaman 'Upload File' untuk mengakses halaman ini.")

# -----------------------------------------------------------------------------
# Page Implementations
# -----------------------------------------------------------------------------

def page_beranda():
    st.header("Selamat Datang di Aplikasi Prediksi IHK")
    st.markdown("Aplikasi ini menggunakan Radial Basis Function Neural Network (RBFNN) untuk melakukan prediksi Indeks Harga Konsumen (IHK). Navigasikan melalui menu di atas untuk memulai.")
    
    st.subheader("Apa itu RBFNN?")
    col1, col2, col3 = st.columns([1,2,1]) # Buat 3 kolom, kolom tengah lebih lebar

    with col2: # Masukkan gambar di kolom tengah
        st.image("Arsitektur RBFNN.png", caption="Arsitektur Jaringan RBFNN", width=600)
    st.markdown("""
    Radial Basis Function Neural Network (RBFNN) merupakan salah satu arsitektur Jaringan Syaraf Tiruan (JST) yang bersifat feedforward. RBFNN menggunakan fungsi aktivasi berbasis radial, yaitu fungsi yang nilainya bergantung pada jarak antara suatu titik input dengan titik pusat tertentu di dalam ruang input. Arsitektur RBFNN memiliki tiga lapisan:
    1.  **Lapisan Input (Input Layer):** Menerima data input, yang dapat ditentukan menggunakan uji PACF untuk memilih variabel input yang relevan.
    2.  **Lapisan Tersembunyi (Hidden Layer):** Terdiri dari neuron RBF yang menghitung fungsi aktivasi berdasarkan jarak antara input dan pusat neuron.
    3.  **Lapisan Output (Output Layer):** Menghasilkan output prediksi dari hasil perkalian antara bobot dengan fungsi aktivasi.
    """)
    


def page_upload_file():
    st.header("📤 Upload File Data IHK")
    st.info("ℹ️ File harus berformat CSV, univariat, dan memiliki kolom 'Tanggal' serta kolom nilai.")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("🎉 File berhasil diupload!")
            
            st.subheader("Pilih Kolom Tanggal dan Data")
            available_columns = data.columns.tolist()
            col1, col2 = st.columns(2)
            date_col = col1.selectbox("Pilih Kolom Tanggal:", available_columns, index=available_columns.index(st.session_state.date_column_name) if st.session_state.date_column_name in available_columns else 0)
            value_col = col2.selectbox("Pilih Kolom Data (Nilai IHK):", available_columns, index=available_columns.index(st.session_state.data_column_name) if st.session_state.data_column_name in available_columns else 1 if len(available_columns)>1 else 0)

            st.session_state.date_column_name = date_col
            st.session_state.data_column_name = value_col
            
            data_timeseries = data[[date_col, value_col]].copy()
            data_timeseries[date_col] = pd.to_datetime(data_timeseries[date_col])
            data_timeseries.rename(columns={value_col: 'IHK_Value', date_col: 'Tanggal'}, inplace=True)
            data_timeseries.set_index('Tanggal', inplace=True)
            data_timeseries = data_timeseries[['IHK_Value']]

            st.session_state.data_timeseries = data_timeseries
            st.session_state.file_uploaded = True
            st.session_state.normalizer = DataPreprocessing(data_column_name='IHK_Value')

            st.subheader("Pratinjau Data (5 Baris Pertama)")
            st.dataframe(st.session_state.data_timeseries.head())

            st.subheader("Plot Data Time Series")
            fig, ax = plt.subplots(figsize=(12, 6), facecolor=CONTENT_BG_COLOR) # Gunakan warna tema
            ax.plot(st.session_state.data_timeseries.index, st.session_state.data_timeseries['IHK_Value'], color=PLOT_LINE_COLOR, linewidth=2)
            ax.set_title(f'Data {st.session_state.data_column_name} Historis', color=TEXT_COLOR)
            ax.set_xlabel('Tanggal', color=TEXT_COLOR)
            ax.set_ylabel(st.session_state.data_column_name, color=TEXT_COLOR)
            ax.grid(True, linestyle='--', alpha=0.3, color=PLOT_GRID_COLOR) # Warna grid
            ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
            ax.set_facecolor(CONTENT_BG_COLOR) # Latar belakang axis
            for spine in ax.spines.values(): # Warna bingkai
                spine.set_edgecolor(PLOT_GRID_COLOR)
            st.pyplot(fig)

            st.subheader("Analisis Deskriptif")
            st.dataframe(st.session_state.data_timeseries['IHK_Value'].describe().to_frame().T)
            
            if st.button("Lanjut ke Pre-processing ➡️", key="upload_next"):
                st.session_state.current_page = "Pre-processing"
                st.rerun()
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
            st.session_state.file_uploaded = False

def page_preprocessing():
    st.header("⚙️ Pre-processing Data")
    if st.session_state.data_timeseries is None:
        st.warning("⛔ Silakan upload data terlebih dahulu di halaman 'Upload File'.")
        return

    data_ts = st.session_state.data_timeseries
    normalizer_obj = st.session_state.normalizer

    st.subheader("1. Pengecekan Missing Values")
    total_missing, missing_details = normalizer_obj.check_missing_values(data_ts)
    if total_missing > 0:
        st.warning(f"⚠️ Terdapat {total_missing} missing value pada data:")
        st.write(missing_details)
        data_ts_imputed = data_ts.fillna(method='ffill').fillna(method='bfill')
        if data_ts_imputed.isnull().sum().sum() > 0:
            st.error("Imputasi gagal menghilangkan semua missing values.")
            return
        else:
            st.success("Missing values diimputasi menggunakan forward & backward fill.")
            st.session_state.data_timeseries = data_ts_imputed
            data_ts = data_ts_imputed
    else:
        st.success("✅ Tidak terdapat missing value pada data.")

    st.subheader("2. Normalisasi Data")
    try:
        normalizer_obj.fit_scaler(data_ts)
        normalized_values = normalizer_obj.normalize(data_ts[[normalizer_obj.data_column_name]])
        
        normalized_df_display = data_ts.copy()
        normalized_df_display['Data Normalisasi'] = normalized_values
        normalized_df_display.rename(columns={normalizer_obj.data_column_name: 'Data Asli'}, inplace=True)
        
        st.markdown("Data dinormalisasi ke rentang [0, 1]. Berikut adalah contohnya:")
        st.dataframe(normalized_df_display[['Data Asli', 'Data Normalisasi']].head())

        st.session_state.normalized_series = pd.Series(normalized_values.flatten(), index=data_ts.index)
        st.session_state.normalizer = normalizer_obj
        
        st.subheader("3. Identifikasi Lag Signifikan (PACF)")
        st.markdown("""
        Plot PACF (Partial Autocorrelation Function) membantu mengidentifikasi lag yang signifikan 
        untuk model time series. Lag yang signifikan pada PACF sering digunakan sebagai input untuk model.
        """)
        
        lags_to_plot = st.slider("Jumlah lag untuk plot PACF:", min_value=10, max_value=min(60, len(st.session_state.normalized_series)//3), value=24)
        fig_pacf = plot_pacf_only_streamlit(st.session_state.normalized_series, lags=lags_to_plot, series_name="Data Ternormalisasi")
        st.pyplot(fig_pacf)

        st.markdown("Masukkan lag signifikan berdasarkan plot PACF (pisahkan dengan koma):")
        lags_input_str = st.text_input("Lag Signifikan:", st.session_state.significant_lags_input, key="lags_input_preprocess")
        st.session_state.significant_lags_input = lags_input_str

        if st.button("Lanjut ke Pemodelan ➡️", key="preprocess_next"):
            try:
                if not [int(lag.strip()) for lag in lags_input_str.split(',') if lag.strip()]:
                    st.error("Harap masukkan setidaknya satu lag yang valid.")
                else:
                    st.session_state.current_page = "Pemodelan"
                    st.rerun()
            except ValueError:
                st.error("Format lag tidak valid. Harap masukkan angka yang dipisahkan koma.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat normalisasi atau plot PACF: {e}")

def page_pemodelan():
    st.header("🛠️ Pemodelan RBFNN")
    if st.session_state.normalized_series is None:
        st.warning("⛔ Data belum diproses. Silakan lengkapi tahap 'Pre-processing'.")
        return

    st.sidebar.header("⚙️ Parameter Model RBFNN")
    activation_fn = st.sidebar.selectbox("Fungsi Aktivasi:", ['gaussian', 'multiquadric'], key="activation_select")
    split_ratio_str = st.sidebar.selectbox("Rasio Data Training:Testing:", ["80:20", "70:30", "90:10"], index=0, key="split_ratio_select")
    train_split_ratio = {"90:10": 0.9, "80:20": 0.8, "70:30": 0.7}[split_ratio_str]
    
    num_centers = st.sidebar.radio("Jumlah Center (Neuron RBF):", [1, 2, 3, 4, 5], index=2, horizontal=True, key="centers_radio")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Lag Input Model")
    lags_input_display = st.sidebar.text_input("Lag yang Digunakan:", st.session_state.significant_lags_input, key="lags_display_model")
    try:
        significant_lags = [int(lag.strip()) for lag in lags_input_display.split(',') if lag.strip()]
        if not significant_lags:
            st.error("Lag signifikan tidak valid atau kosong.")
            return
        st.sidebar.info(f"Model akan menggunakan lag: {significant_lags}")
    except ValueError:
        st.error("Format lag tidak valid.")
        return
    
    st.subheader("Konfigurasi Model Saat Ini")
    col_conf1, col_conf2, col_conf3 = st.columns(3)
    col_conf1.metric("Fungsi Aktivasi", activation_fn.capitalize())
    col_conf2.metric("Rasio Split (Train)", f"{int(train_split_ratio*100)}%")
    col_conf3.metric("Jumlah Center", num_centers)
    st.markdown(f"**Lag Input yang Digunakan:** `{significant_lags}`")

    if st.button("Latih Model RBFNN 🚀", key="train_model_button"):
        with st.spinner("Sedang melatih model RBFNN... Mohon tunggu 🙏"):
            try:
                X_all, y_all, idx_all = create_supervised_data(st.session_state.normalized_series, significant_lags)
                if X_all.shape[0] == 0:
                    st.error("Gagal membuat data supervised. Lag terlalu besar atau data terlalu sedikit.")
                    return

                split_index = int(train_split_ratio * len(X_all))
                X_train, X_test = X_all[:split_index], X_all[split_index:]
                y_train, y_test = y_all[:split_index], y_all[split_index:]
                idx_train, idx_test = idx_all[:split_index], idx_all[split_index:]

                if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                    st.error("Pembagian data gagal. Data training atau testing kosong.")
                    return
                
                rbf_model = RBFNN(n_centers=num_centers, activation=activation_fn)
                if not rbf_model.fit(X_train, y_train):
                    st.session_state.trained_rbf_model = None
                    return
                
                st.session_state.trained_rbf_model = rbf_model
                st.session_state.model_params = {'activation': activation_fn, 'split_ratio': train_split_ratio, 'n_centers': num_centers, 'lags': significant_lags, 'full_normalized_series_for_pred': st.session_state.normalized_series}
                st.success("✅ Model RBFNN berhasil dilatih!")

                st.subheader("📈 Hasil Pelatihan & Evaluasi Model")
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("Nilai Spread (σ) Model", f"{rbf_model.spread:.4f}")

                y_train_pred_norm, y_test_pred_norm = rbf_model.predict(X_train), rbf_model.predict(X_test)
                normalizer = st.session_state.normalizer
                y_train_denorm, y_train_pred_denorm = normalizer.denormalize(y_train.reshape(-1, 1)), normalizer.denormalize(y_train_pred_norm)
                y_test_denorm, y_test_pred_denorm = normalizer.denormalize(y_test.reshape(-1, 1)), normalizer.denormalize(y_test_pred_norm)
                
                train_smape_val, test_smape_val = smape(y_train_denorm, y_train_pred_denorm), smape(y_test_denorm, y_test_pred_denorm)
                col_res1.metric("Train SMAPE", f"{train_smape_val:.2f}%")
                col_res2.metric("Test SMAPE", f"{test_smape_val:.2f}%")
                
                st.subheader("Visualisasi Hasil Prediksi (Data Asli)")
                fig_pred, ax_pred = plt.subplots(figsize=(14, 7), facecolor=CONTENT_BG_COLOR) # Gunakan warna tema
                full_original_data = st.session_state.data_timeseries[st.session_state.normalizer.data_column_name]
                ax_pred.plot(full_original_data.index, full_original_data.values, label='Data Asli (Historis)', color=PLOT_GRID_COLOR, alpha=0.7, linestyle='--')
                ax_pred.plot(idx_train, y_train_denorm, label='Data Training (Asli)', color='#3182CE', marker='.', linestyle='') # Biru cerah
                ax_pred.plot(idx_train, y_train_pred_denorm, label='Prediksi Training', color=PLOT_LINE_COLOR, linestyle='--') # Biru aksen
                ax_pred.plot(idx_test, y_test_denorm, label='Data Testing (Asli)', color='#38A169', marker='.', linestyle='') # Hijau cerah
                ax_pred.plot(idx_test, y_test_pred_denorm, label='Prediksi Testing', color='#68D391', linestyle='--') # Hijau lebih terang
                
                ax_pred.set_title(f'Perbandingan Data Asli vs Prediksi ({st.session_state.data_column_name})', color=TEXT_COLOR)
                ax_pred.set_xlabel('Tanggal', color=TEXT_COLOR)
                ax_pred.set_ylabel(st.session_state.data_column_name, color=TEXT_COLOR)
                ax_pred.legend(facecolor=CONTENT_BG_COLOR, edgecolor=PLOT_GRID_COLOR, labelcolor=TEXT_COLOR) # Legend dengan tema
                ax_pred.grid(True, linestyle='--', alpha=0.3, color=PLOT_GRID_COLOR)
                ax_pred.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
                ax_pred.set_facecolor(CONTENT_BG_COLOR) # Latar belakang axis
                for spine in ax_pred.spines.values(): # Warna bingkai
                    spine.set_edgecolor(PLOT_GRID_COLOR)
                st.pyplot(fig_pred)

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat melatih model: {e}")
                import traceback
                st.error(traceback.format_exc())
                st.session_state.trained_rbf_model = None

    if st.session_state.trained_rbf_model is not None:
        st.markdown("---")
        if st.button("Lanjut ke Prediksi Masa Depan 🔮", key="model_next"):
            st.session_state.current_page = "Prediksi"
            st.rerun()

def page_prediksi():
    st.header("🔮 Prediksi IHK 5 Bulan ke Depan")

    if st.session_state.trained_rbf_model is None:
        st.warning("⛔ Model belum dilatih. Silakan latih model di halaman 'Pemodelan'.")
        return

    model_to_use = st.session_state.trained_rbf_model
    params_used = st.session_state.model_params
    normalizer = st.session_state.normalizer
    
    st.subheader("Menggunakan Model yang Dilatih dengan Parameter:")
    param_df_data = {"Parameter": ["Fungsi Aktivasi", "Rasio Split", "Jumlah Center", "Lag Input"], "Nilai": [params_used['activation'].capitalize(), f"{int(params_used['split_ratio']*100)}%", params_used['n_centers'], str(params_used['lags'])]}
    st.table(pd.DataFrame(param_df_data))

    with st.spinner("Melakukan prediksi 5 bulan ke depan..."):
        try:
            full_normalized_series = params_used['full_normalized_series_for_pred']
            max_lag = max(params_used['lags'])
            if len(full_normalized_series) < max_lag:
                st.error(f"Data historis tidak cukup untuk lag terbesar ({max_lag}).")
                return

            initial_data_norm = full_normalized_series[-max_lag:].values
            future_predictions_denorm = predict_future(model=model_to_use, initial_data_normalized=initial_data_norm, lags=params_used['lags'], steps=5, normalizer=normalizer)
            
            last_date_historis = st.session_state.data_timeseries.index[-1]
            future_dates = pd.date_range(start=last_date_historis + pd.DateOffset(months=1), periods=5, freq='MS')
            
            future_df = pd.DataFrame({'Tanggal': future_dates, f'Prediksi {st.session_state.data_column_name}': future_predictions_denorm})
            
            st.subheader("Hasil Prediksi 5 Bulan ke Depan:")
            st.dataframe(future_df.style.format({'Tanggal': lambda t: t.strftime('%Y-%m-%d'), f'Prediksi {st.session_state.data_column_name}': "{:.2f}"}))

            st.subheader("Visualisasi Prediksi vs Data Historis")
            fig_future, ax_future = plt.subplots(figsize=(14, 7), facecolor=CONTENT_BG_COLOR) # Gunakan warna tema
            original_data_series = st.session_state.data_timeseries[normalizer.data_column_name]
            ax_future.plot(original_data_series.index, original_data_series.values, label='Data Historis Asli', color=PLOT_LINE_COLOR)
            ax_future.plot(future_dates, future_predictions_denorm, label='Prediksi 5 Bulan ke Depan', color='#F56565', marker='o', linestyle='--') # Merah untuk prediksi
            
            ax_future.set_title(f'Prediksi {st.session_state.data_column_name} 5 Bulan ke Depan', color=TEXT_COLOR)
            ax_future.set_xlabel('Tanggal', color=TEXT_COLOR)
            ax_future.set_ylabel(st.session_state.data_column_name, color=TEXT_COLOR)
            ax_future.legend(facecolor=CONTENT_BG_COLOR, edgecolor=PLOT_GRID_COLOR, labelcolor=TEXT_COLOR) # Legend dengan tema
            ax_future.grid(True, linestyle='--', alpha=0.3, color=PLOT_GRID_COLOR)
            ax_future.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
            ax_future.set_facecolor(CONTENT_BG_COLOR) # Latar belakang axis
            for spine in ax_future.spines.values(): # Warna bingkai
                spine.set_edgecolor(PLOT_GRID_COLOR)
            st.pyplot(fig_future)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi masa depan: {e}")
            import traceback
            st.error(traceback.format_exc())

    st.markdown("---")
    if st.button("↩️ Upload Data Baru untuk Prediksi Lain", key="predict_back"):
        st.session_state.data_timeseries = None
        st.session_state.file_uploaded = False
        st.session_state.normalized_series = None
        st.session_state.trained_rbf_model = None
        st.session_state.model_params = {}
        st.session_state.current_page = "Upload File"
        st.rerun()

# -----------------------------------------------------------------------------
# Main App Logic: Page Routing
# -----------------------------------------------------------------------------
if st.session_state.current_page == "Beranda":
    page_beranda()
elif st.session_state.current_page == "Upload File":
    page_upload_file()
elif st.session_state.current_page == "Pre-processing":
    if st.session_state.file_uploaded:
        page_preprocessing()
elif st.session_state.current_page == "Pemodelan":
    if st.session_state.file_uploaded:
        page_pemodelan()
elif st.session_state.current_page == "Prediksi":
    if st.session_state.file_uploaded:
        page_prediksi()

st.markdown("---")
st.markdown(f"<p style='text-align: center; font-size: small; color: {ACCENT_COLOR};'>Aplikasi Prediksi IHK - Dibuat dengan Streamlit</p>", unsafe_allow_html=True)
