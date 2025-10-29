"""
Streamlit EEG Processor
Single-file Streamlit app to upload multiple EEG files (14 channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4),
apply filtering/normalization/segmentation (with overlap), extract time / frequency / time-frequency features,
compute Mutual Information (MI) scores (requires labels) and save most significant features to CSV.

How it works (UI):
- Upload one or more CSV/XLSX files containing raw EEG columns (14 columns in the channel order above).
- Optionally upload a label CSV that maps filename -> label (two columns: filename, label)
  OR choose "Infer labels from filename (regex)" and provide a regex capture group.
- Choose preprocessing: filter type & params, normalization toggle.
- Choose segmentation: window length (seconds), overlap fraction (0-0.99), sampling rate.
- Extract features, calculate MI (if labels available), display feature table and top features.
- Pick MI threshold and save selected features to CSV.

Save this file to a GitHub repo and run with: streamlit run streamlit_eeg_processor.py

Dependencies: streamlit, numpy, pandas, scipy, scikit-learn, pywt, openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
import pywt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import NotFittedError

# ----------------------------- Helper functions ---------------------------------
CHANNEL_ORDER = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]

def read_eeg_file(file_bytes, filename):
    # Accept CSV or Excel
    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
    except Exception as e:
        raise ValueError(f"Failed to read {filename}: {e}")

    # If dataframe has more columns, try to select first 14
    if df.shape[1] < 14:
        raise ValueError(f"File {filename} has fewer than 14 columns ({df.shape[1]}). Expected 14 channels.")

    df = df.iloc[:, :14].copy()
    df.columns = CHANNEL_ORDER
    return df

# Filtering
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a

def apply_filter(df, fs, filter_type, params):
    out = df.copy()
    for ch in df.columns:
        sig = df[ch].values.astype(float)
        if filter_type == 'None':
            out[ch] = sig
        elif filter_type == 'Bandpass':
            b,a = butter_bandpass(params['low'], params['high'], fs, order=params['order'])
            out[ch] = filtfilt(b,a,sig)
        elif filter_type == 'Lowpass':
            b,a = butter_lowpass(params['cutoff'], fs, order=params['order'])
            out[ch] = filtfilt(b,a,sig)
        elif filter_type == 'Highpass':
            b,a = butter_highpass(params['cutoff'], fs, order=params['order'])
            out[ch] = filtfilt(b,a,sig)
    return out

# Segmentation
def segment_signal(df, fs, window_sec, overlap_frac):
    window_samples = int(window_sec * fs)
    step = int(window_samples * (1 - overlap_frac))
    if window_samples <= 0 or step <= 0:
        raise ValueError("window or overlap leads to non-positive step/window size")
    segments = []  # list of (start_idx, end_idx, dataframe segment)
    n = df.shape[0]
    for start in range(0, n - window_samples + 1, step):
        end = start + window_samples
        seg = df.iloc[start:end].reset_index(drop=True)
        segments.append((start, end, seg))
    return segments

# Feature extraction
def time_features(sig):
    # sig: 1D numpy array
    features = {}
    features['mean'] = np.mean(sig)
    features['std'] = np.std(sig)
    features['skew'] = skew(sig)
    features['kurtosis'] = kurtosis(sig)
    features['rms'] = np.sqrt(np.mean(sig**2))
    features['median'] = np.median(sig)
    # zero crossings
    zc = ((sig[:-1] * sig[1:]) < 0).sum()
    features['zero_crossings'] = int(zc)
    # Hjorth
    first_der = np.diff(sig)
    second_der = np.diff(sig, n=2)
    var0 = np.var(sig)
    var1 = np.var(first_der) if first_der.size>0 else 0.0
    var2 = np.var(second_der) if second_der.size>0 else 0.0
    features['hjorth_activity'] = var0
    features['hjorth_mobility'] = np.sqrt(var1/var0) if var0>0 else 0.0
    features['hjorth_complexity'] = np.sqrt(var2/var1)/features['hjorth_mobility'] if var1>0 and features['hjorth_mobility']>0 else 0.0
    # spectral entropy later
    return features

def frequency_features(sig, fs):
    features = {}
    f, Pxx = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    # band power delta(1-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-50)
    bands = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}
    total_power = np.trapz(Pxx, f)
    for bname,(low,high) in bands.items():
        idx = np.logical_and(f>=low, f<high)
        band_power = np.trapz(Pxx[idx], f[idx]) if idx.any() else 0.0
        features[f'pow_{bname}'] = band_power
        features[f'rel_pow_{bname}'] = band_power / total_power if total_power>0 else 0.0

    # spectral entropy
    ps = Pxx / (Pxx.sum() + 1e-12)
    spectral_entropy = -np.sum(ps * np.log(ps + 1e-12))
    features['spectral_entropy'] = spectral_entropy
    # peak frequency
    features['peak_freq'] = f[np.argmax(Pxx)] if len(f)>0 else 0.0
    return features

def timefreq_features(sig, fs, wavelet='db4', level=4):
    features = {}
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    # energy of each detail approx coefficients
    for i, c in enumerate(coeffs):
        features[f'wt_energy_{i}'] = np.sum(np.square(c))
        features[f'wt_mean_{i}'] = np.mean(c)
    return features

# Combine feature extraction for a segment (all channels)
def extract_features_from_segment(seg_df, fs):
    feat = {}
    for ch in seg_df.columns:
        sig = seg_df[ch].values.astype(float)
        tfeat = time_features(sig)
        ffeat = frequency_features(sig, fs)
        wfeat = timefreq_features(sig, fs)
        # prefix channel
        for k,v in {**tfeat, **ffeat, **wfeat}.items():
            feat[f"{ch}_{k}"] = v
    return feat

# Normalize dataframe features
def normalize_features(df_features, method='MinMax'):
    if method == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    cols = df_features.columns
    scaled = scaler.fit_transform(df_features.values)
    return pd.DataFrame(scaled, columns=cols, index=df_features.index)

# Compute MI
def compute_mi(X_df, y_series):
    # X_df: features, y_series: labels (discrete)
    try:
        mi = mutual_info_classif(X_df.values, y_series.values, discrete_features=False, random_state=0)
    except Exception as e:
        raise RuntimeError(f"mutual_info_classif failed: {e}")
    return pd.Series(mi, index=X_df.columns)

# ----------------------------- Streamlit UI ---------------------------------
st.set_page_config(page_title='EEG Processor', layout='wide')
st.title('EEG Batch Processor — Streamlit')
st.markdown('Upload multiple EEG files (CSV/XLSX). Each file must have 14 columns in the channel order: ' + ', '.join(CHANNEL_ORDER))

# Sidebar config
st.sidebar.header('Preprocessing')
sampling_rate = st.sidebar.number_input('Sampling rate (Hz)', value=128, step=1)
filter_type = st.sidebar.selectbox('Filter type', ['None','Bandpass','Lowpass','Highpass'])
filter_order = st.sidebar.slider('Filter order', 1, 8, 4)
filter_params = {}
if filter_type == 'Bandpass':
    low = st.sidebar.number_input('Bandpass low (Hz)', value=1.0, step=0.1)
    high = st.sidebar.number_input('Bandpass high (Hz)', value=50.0, step=0.1)
    filter_params['low'] = low; filter_params['high'] = high; filter_params['order'] = filter_order
elif filter_type in ['Lowpass','Highpass']:
    cutoff = st.sidebar.number_input('Cutoff (Hz)', value=50.0, step=0.1)
    filter_params['cutoff'] = cutoff; filter_params['order'] = filter_order

normalize = st.sidebar.selectbox('Normalization', ['None','MinMax','Standard'])

st.sidebar.header('Segmentation')
window_sec = st.sidebar.number_input('Window length (seconds)', value=2.0, step=0.1)
overlap = st.sidebar.slider('Overlap fraction', 0.0, 0.95, 0.5, step=0.05)

st.sidebar.header('MI / Feature selection')
label_mode = st.sidebar.selectbox('Label mode', ['No labels','Upload label CSV','Infer from filename (regex)'])
label_file = None
label_regex = ''
if label_mode == 'Upload label CSV':
    label_file = st.sidebar.file_uploader('Upload label CSV (filename,label)', type=['csv','xlsx'], key='label')
elif label_mode == 'Infer from filename (regex)':
    label_regex = st.sidebar.text_input('Regex with one capture group (e.g. (happy|sad) )', value='')

mi_threshold = st.sidebar.slider('MI threshold (keep feature if MI >= threshold)', 0.0, 1.0, 0.1, step=0.01)

# File upload
st.header('Files')
uploaded_files = st.file_uploader('Upload CSV / XLSX files (multi)', accept_multiple_files=True, type=['csv','xlsx'])

if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded")

process_button = st.button('Process files')

# Container for results
results_container = st.container()

if process_button:
    if not uploaded_files:
        st.error('Please upload one or more EEG files first.')
    else:
        all_features = []
        file_labels = {}
        # load labels if provided
        if label_mode == 'Upload label CSV' and label_file is not None:
            try:
                if label_file.name.lower().endswith('.csv'):
                    labels_df = pd.read_csv(label_file)
                else:
                    labels_df = pd.read_excel(label_file, engine='openpyxl')
                # expect columns filename,label
                if labels_df.shape[1] < 2:
                    st.warning('Label file must have at least two columns: filename,label')
                else:
                    labels_df = labels_df.iloc[:, :2]
                    labels_df.columns = ['filename','label']
                    for _,r in labels_df.iterrows():
                        file_labels[str(r['filename'])] = r['label']
            except Exception as e:
                st.warning(f'Could not read label file: {e}')

        for f in uploaded_files:
            try:
                df = read_eeg_file(f.read(), f.name)
            except Exception as e:
                st.error(str(e))
                continue

            # preprocessing
            if filter_type != 'None':
                try:
                    df = apply_filter(df, sampling_rate, filter_type, filter_params)
                except Exception as e:
                    st.warning(f'Filtering failed for {f.name}: {e}')
            if normalize != 'None':
                if normalize == 'MinMax':
                    df = pd.DataFrame(MinMaxScaler().fit_transform(df.values), columns=df.columns)
                else:
                    df = pd.DataFrame(StandardScaler().fit_transform(df.values), columns=df.columns)

            # segmentation
            segments = segment_signal(df, sampling_rate, window_sec, overlap)
            st.write(f'File {f.name}: {len(segments)} segments created (window {window_sec}s, overlap {overlap})')

            # extract features per segment and attach metadata
            for (start,end,seg) in segments:
                feat = extract_features_from_segment(seg, sampling_rate)
                feat['file'] = f.name
                feat['start'] = start
                feat['end'] = end
                # attach label if available
                label_val = None
                if label_mode == 'Upload label CSV':
                    label_val = file_labels.get(f.name, None)
                elif label_mode == 'Infer from filename (regex)' and label_regex:
                    import re
                    m = re.search(label_regex, f.name)
                    if m:
                        label_val = m.group(1)
                feat['label'] = label_val
                all_features.append(feat)

        if not all_features:
            st.error('No features were extracted.')
        else:
            feats_df = pd.DataFrame(all_features)
            # move metadata/label columns to front
            meta_cols = ['file','start','end','label']
            other_cols = [c for c in feats_df.columns if c not in meta_cols]
            feats_df = feats_df[meta_cols + other_cols]

            st.subheader('Extracted features (preview)')
            st.dataframe(feats_df.head(200))

            # compute MI if labels available
            if feats_df['label'].notnull().any():
                st.success('Labels detected — computing MI scores')
                labeled_df = feats_df[feats_df['label'].notnull()].copy()
                X = labeled_df.drop(columns=meta_cols)
                y = labeled_df['label'].astype(str)
                try:
                    mi_series = compute_mi(X, y)
                    mi_df = mi_series.sort_values(ascending=False).rename_axis('feature').reset_index(name='mi')
                    st.subheader('Top features by MI')
                    st.dataframe(mi_df.head(50))

                    # join MI back to features
                    mi_map = mi_series.to_dict()
                    for col in X.columns:
                        feats_df[f'{col}__mi'] = mi_map.get(col,0.0)

                    # selection by threshold
                    kept_features = mi_df[mi_df['mi'] >= mi_threshold]['feature'].tolist()
                    st.write(f'{len(kept_features)} features pass MI threshold {mi_threshold}')

                    # Show option to save
                    if st.checkbox('Show features passing threshold'):
                        st.dataframe(mi_df[mi_df['mi'] >= mi_threshold])

                    if st.button('Save selected features to CSV'):
                        if not kept_features:
                            st.warning('No features to save with current threshold.')
                        else:
                            # create CSV with metadata + kept features
                            save_cols = meta_cols + kept_features
                            out_df = feats_df[save_cols].copy()
                            csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                            st.download_button('Download CSV', data=csv_bytes, file_name='selected_features.csv', mime='text/csv')

                except Exception as e:
                    st.error(f'Failed to compute MI: {e}')
            else:
                st.warning('No labels detected: Mutual Information cannot be computed without labels.')
                # fallback: compute variance-based ranking
                var_rank = feats_df.drop(columns=meta_cols).var().sort_values(ascending=False).rename_axis('feature').reset_index(name='variance')
                st.subheader('Feature ranking by variance (no labels available)')
                st.dataframe(var_rank.head(50))
                if st.button('Save top-variance features to CSV'):
                    topf = var_rank['feature'].head(100).tolist()
                    out_df = feats_df[meta_cols + topf]
                    csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                    st.download_button('Download CSV', data=csv_bytes, file_name='top_variance_features.csv', mime='text/csv')

        st.success('Processing complete.')

st.markdown('---')
st.caption('Export instructions: run `pip install -r requirements.txt` then `streamlit run streamlit_eeg_processor.py`.\nCreate requirements.txt with: streamlit, numpy, pandas, scipy, scikit-learn, pywt, openpyxl')
