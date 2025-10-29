import streamlit as st
import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis, entropy
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
import pywt

# ======================================================
# Streamlit EEG Feature Extraction App
# ======================================================

st.set_page_config(page_title="EEG Feature Extraction", layout="wide")

st.title("🧠 EEG Feature Extraction & Mutual Information Analysis")

# --- Channel Order ---
channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# ======================================================
# Sidebar Controls
# ======================================================
st.sidebar.header("🔧 Processing Settings")

filter_type = st.sidebar.selectbox(
    "Filter Type",
    ["None", "Low-pass", "High-pass", "Band-pass"]
)

normalize = st.sidebar.checkbox("Apply Normalization", True)
segment_length = st.sidebar.number_input("Segment length (samples)", 256, 4096, 512, 128)
overlap = st.sidebar.slider("Overlap (%)", 0, 90, 50)

mi_threshold = st.sidebar.slider("Mutual Information Threshold", 0.0, 1.0, 0.05, 0.01)

# ======================================================
# File Upload
# ======================================================
uploaded_files = st.file_uploader(
    "Upload multiple EEG files (CSV or XLSX)", accept_multiple_files=True)

if uploaded_files:
    all_features = []
    all_labels = []

    for file in uploaded_files:
        # --- Read File ---
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.write(f"📄 Processing: `{file.name}` ({len(df)} samples)")
        df = df[channels]  # Reorder columns if necessary

        data = df.values.T  # shape: (14, N)
        fs = 128  # Hz

        # ======================================================
        # Filtering
        # ======================================================
        def butter_filter(sig, low=1, high=40, fs=128, btype='band'):
            nyq = 0.5 * fs
            low /= nyq
            high /= nyq
            b, a = butter(4, [low, high], btype=btype)
            return filtfilt(b, a, sig)

        filtered_data = []
        for ch in data:
            if filter_type == "Low-pass":
                filtered = butter_filter(ch, high=30, low=0.1, fs=fs, btype='low')
            elif filter_type == "High-pass":
                filtered = butter_filter(ch, high=fs/2 - 1, low=1, fs=fs, btype='high')
            elif filter_type == "Band-pass":
                filtered = butter_filter(ch, low=1, high=40, fs=fs, btype='band')
            else:
                filtered = ch
            filtered_data.append(filtered)

        filtered_data = np.array(filtered_data)

        # ======================================================
        # Segmentation
        # ======================================================
        seg_step = int(segment_length * (1 - overlap / 100))
        num_segments = (filtered_data.shape[1] - segment_length) // seg_step + 1

        feature_list = []
        for i in range(num_segments):
            start = i * seg_step
            end = start + segment_length
            seg = filtered_data[:, start:end]

            # --- Feature extraction ---
            features = []
            for ch_data in seg:
                # Time-domain
                features += [
                    np.mean(ch_data),
                    np.std(ch_data),
                    skew(ch_data),
                    kurtosis(ch_data),
                    np.median(ch_data),
                    np.max(ch_data) - np.min(ch_data)
                ]

                # Frequency-domain
                f, psd = welch(ch_data, fs=fs, nperseg=256)
                features += [
                    np.mean(psd), np.std(psd),
                    np.sum(psd), entropy(psd)
                ]

                # Time-frequency (Wavelet)
                coeffs = pywt.wavedec(ch_data, 'db4', level=3)
                energies = [np.sum(np.square(c)) for c in coeffs]
                features += energies

            feature_list.append(features)

        feature_matrix = np.array(feature_list)

        # ======================================================
        # Normalization
        # ======================================================
        if normalize:
            scaler = MinMaxScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)

        all_features.append(feature_matrix)

        # label inferred from file name
        label = os.path.splitext(file.name)[0]
        all_labels += [label] * feature_matrix.shape[0]

    X = np.vstack(all_features)
    y = np.array(all_labels)

    st.success(f"✅ Extracted {X.shape[0]} segments × {X.shape[1]} features")

    # ======================================================
    # Mutual Information
    # ======================================================
    st.subheader("📊 Mutual Information Analysis")

    mi = mutual_info_classif(X, pd.factorize(y)[0], random_state=0)
    mi_df = pd.DataFrame({"Feature": range(len(mi)), "MI_Score": mi})
    st.dataframe(mi_df.sort_values("MI_Score", ascending=False))

    # Filter features by threshold
    selected_idx = np.where(mi >= mi_threshold)[0]
    X_selected = X[:, selected_idx]

    st.info(f"{len(selected_idx)} features selected (MI ≥ {mi_threshold})")

    # ======================================================
    # Save Option
    # ======================================================
    if st.button("💾 Save Results to CSV"):
        result_df = pd.DataFrame(X_selected)
        result_df["Label"] = y
        result_df.to_csv("EEG_Features_Selected.csv", index=False)
        st.success("Saved as EEG_Features_Selected.csv")

else:
    st.info("Upload EEG files to begin processing.")
