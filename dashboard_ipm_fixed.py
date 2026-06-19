import warnings
warnings.filterwarnings("ignore")

import os, math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import streamlit as st
from scipy import stats


# PAGE CONFIG
st.set_page_config(
    page_title="Dashboard IPM Jawa Timur",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2e6da4 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.5rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; font-size: 0.85rem; opacity: 0.85; }

    .metric-card {
        background: white;
        border: 1px solid #e0e7ef;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .metric-card .val  { font-size: 2rem; font-weight: 700; color: #2e6da4; }
    .metric-card .lbl  { font-size: 0.8rem; color: #666; margin-top: 0.2rem; }
    .metric-card .sub  { font-size: 0.75rem; color: #999; }

    .section-header {
        font-size: 1.05rem; font-weight: 700;
        color: #1e3a5f; border-left: 4px solid #2e6da4;
        padding-left: 0.8rem; margin: 1.2rem 0 0.8rem;
    }
    [data-testid="stSidebar"] { background: #f0f4f9; }
</style>
""", unsafe_allow_html=True)

# CONSTANTS  (sesuai OlahData_v3_FIXED)
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)

DATA_PATH    = "Dataset_IPM.csv"
GEOJSON_PATH = "jawa_timur_kabkota.geojson"

DEP_VAR = "IPM"

# Semua 11 kandidat awal
ALL_CANDIDATES = [
    "Ln_PDRB_Per_Orang",
    "Tingkat_Kemiskinan",
    "Tingkat_Pengangguran",
    "Angka_Melek_Huruf",
    "Angka_Partisipasi_Murni",
    "Ln_Jumlah_Sekolah",
    "Rasio_Murid_dan_Guru",
    "Sanitasi_Kesehatan",
    "Ln_Kepadatan_Penduduk",
    "Keluhan_Kesehatan",
    "Jumlah_Penduduk",
]

# Hanya Jumlah_Penduduk yang dikecualikan dari regresi panel (VARS_PERHATIAN v3)
VARS_PERHATIAN = ["Jumlah_Penduduk"]
INDEP_VARS = [v for v in ALL_CANDIDATES if v not in VARS_PERHATIAN]

# RF menggunakan INDEP_VARS (sama dengan regresi panel)
RF_VARS = INDEP_VARS

# Variabel perhatian khusus di feature importance RF (sesuai Tahap 4C.3 notebook)
FI_PERHATIAN = ["Ln_Jumlah_Sekolah", "Jumlah_Penduduk", "Keluhan_Kesehatan"]

TRAIN_YEARS = list(range(2017, 2023))
TEST_YEARS  = [2023, 2024]

LABEL_MAP = {
    "Ln_PDRB_Per_Orang"      : "Ln PDRB Per Orang",
    "Tingkat_Kemiskinan"     : "Tingkat Kemiskinan (%)",
    "Tingkat_Pengangguran"   : "Tingkat Pengangguran (%)",
    "Angka_Melek_Huruf"      : "Angka Melek Huruf (%)",
    "Angka_Partisipasi_Murni": "Angka Partisipasi Murni (%)",
    "Ln_Jumlah_Sekolah"      : "Ln Jumlah Sekolah",
    "Rasio_Murid_dan_Guru"   : "Rasio Murid & Guru",
    "Sanitasi_Kesehatan"     : "Sanitasi Kesehatan (%)",
    "Ln_Kepadatan_Penduduk"  : "Ln Kepadatan Penduduk",
    "Keluhan_Kesehatan"      : "Keluhan Kesehatan (%)",
    "Jumlah_Penduduk"        : "Jumlah Penduduk",
    "IPM"                    : "IPM",
}

PALETTE = px.colors.qualitative.Set2

# DATA LOADING & PREPROCESSING
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Kabupaten / Kota": "Kabupaten_Kota"})
    df.columns = df.columns.str.replace(" ", "_")

    # Kepadatan → numeric (antisipasi spasi)
    df["Kepadatan_Penduduk"] = (
        df["Kepadatan_Penduduk"].astype(str)
        .str.replace(r"\s", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Forward fill Sanitasi_Kesehatan (missing 2024)
    df = df.sort_values(["Kabupaten_Kota", "Tahun"]).reset_index(drop=True)
    df["Sanitasi_Kesehatan"] = df.groupby("Kabupaten_Kota")["Sanitasi_Kesehatan"].ffill()

    # Log transforms
    df["Ln_PDRB_Per_Orang"]      = np.log(df["PDRB_Per_Orang"].replace(0, np.nan))
    df["Ln_Kepadatan_Penduduk"]  = np.log(df["Kepadatan_Penduduk"].replace(0, np.nan))
    df["Ln_Jumlah_Sekolah"]      = np.log(df["Jumlah_Sekolah"].replace(0, np.nan))

    return df


@st.cache_data
def load_geojson():
    with open(GEOJSON_PATH, "r") as f:
        return json.load(f)


@st.cache_resource
def run_models(df):
    """Jalankan semua model: panel (CEM/FEM/REM + uji asumsi + Cluster SE), K-Means, Random Forest, Sintesis."""
    from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from scipy.stats import shapiro, jarque_bera

    YEARS = sorted(df["Tahun"].unique())
    N_ent = df["Kabupaten_Kota"].nunique()
    T_len = len(TRAIN_YEARS)

    # ── Panel data setup ────────────────────────────────────────────────
    df_panel_full = df[["Tahun", "Kabupaten_Kota", DEP_VAR] + INDEP_VARS].copy()
    df_panel_full = df_panel_full.set_index(["Kabupaten_Kota", "Tahun"]).sort_index()
    df_panel_full.index.names = ["entity", "time"]

    df_panel = df_panel_full.loc[
        df_panel_full.index.get_level_values("time").isin(TRAIN_YEARS)
    ].copy()

    df_panel_test = df_panel_full.loc[
        df_panel_full.index.get_level_values("time").isin(TEST_YEARS)
    ].copy()

    fv = " + ".join(INDEP_VARS)

    # ── Fit tiga model panel (SE unadjusted untuk uji pemilihan) ────────
    res_cem = PooledOLS.from_formula(
        f"{DEP_VAR} ~ 1 + {fv}", data=df_panel
    ).fit(cov_type="unadjusted")

    res_fem = PanelOLS.from_formula(
        f"{DEP_VAR} ~ 1 + {fv} + EntityEffects", data=df_panel
    ).fit(cov_type="unadjusted")

    res_rem = RandomEffects.from_formula(
        f"{DEP_VAR} ~ 1 + {fv}", data=df_panel
    ).fit(cov_type="unadjusted")

    # ── Uji Chow ────────────────────────────────────────────────────────
    f_chow = res_fem.f_pooled.stat
    p_chow = res_fem.f_pooled.pval

    # ── Uji Hausman ─────────────────────────────────────────────────────
    b_fe = res_fem.params
    b_re = res_rem.params.reindex(b_fe.index).dropna()
    b_fe_aligned = b_fe.reindex(b_re.index)
    diff = b_fe_aligned - b_re
    cov_diff = (res_fem.cov.loc[b_re.index, b_re.index]
                - res_rem.cov.loc[b_re.index, b_re.index])
    try:
        H = float(diff.values @ np.linalg.inv(cov_diff.values) @ diff.values)
    except np.linalg.LinAlgError:
        H = float(diff.values @ np.linalg.pinv(cov_diff.values) @ diff.values)
    H = max(H, 0)
    p_hausman = 1 - stats.chi2.cdf(H, len(b_re)) if H > 0 else 1.0

    # ── Uji Breusch-Pagan LM ────────────────────────────────────────────
    resids_cem = res_cem.resids
    sum_sq_total   = np.sum(resids_cem ** 2)
    sum_sq_between = np.sum(resids_cem.groupby(level=0).sum() ** 2)
    LM   = (N_ent * T_len / (2 * (T_len - 1))) * ((sum_sq_between / sum_sq_total) - 1) ** 2
    p_lm = 1 - stats.chi2.cdf(LM, 1)

    # ── Pilih model terbaik ─────────────────────────────────────────────
    if p_chow < 0.05:
        if p_hausman < 0.05:
            best_name = "Fixed Effect Model (FEM)"
        else:
            best_name = "Random Effect Model (REM)"
    else:
        if p_lm < 0.05:
            best_name = "Random Effect Model (REM)"
        else:
            best_name = "Common Effect Model (CEM)"

    # ── Uji Asumsi Klasik (sesuai Tahap 4A.3 notebook) ─────────────────
    # Ambil residual dari model terpilih (unadjusted dulu)
    _res_for_diag = {"Fixed Effect Model (FEM)": res_fem,
                     "Random Effect Model (REM)": res_rem,
                     "Common Effect Model (CEM)": res_cem}[best_name]
    resid_arr = _res_for_diag.resids.values.flatten()

    # Normalitas — Jarque-Bera + Shapiro-Wilk
    jb_stat, jb_pval = jarque_bera(resid_arr)
    sw_stat, sw_pval = shapiro(resid_arr[:min(len(resid_arr), 5000)])
    normality_ok = jb_pval >= 0.05

    # Heteroskedastisitas — Levene per entitas
    resid_ent_df = pd.DataFrame({
        "resid" : resid_arr,
        "entity": df_panel.index.get_level_values(0)
    })
    groups_ent = [g["resid"].values for _, g in resid_ent_df.groupby("entity")]
    lev_stat, lev_pval = stats.levene(*groups_ent)
    hetero_detected = lev_pval < 0.05

    # Autokorelasi — Wooldridge AR(1) style
    resid_wt = pd.DataFrame({
        "resid" : resid_arr,
        "entity": df_panel.index.get_level_values(0),
        "time"  : df_panel.index.get_level_values(1)
    }).sort_values(["entity", "time"])
    resid_wt["resid_fd"]   = resid_wt.groupby("entity")["resid"].diff()
    resid_wt["resid_lag1"] = resid_wt.groupby("entity")["resid"].shift(1)
    resid_wt = resid_wt.dropna(subset=["resid_fd", "resid_lag1"])
    r_wd, _ = stats.pearsonr(resid_wt["resid_lag1"].values, resid_wt["resid_fd"].values)
    n_wd    = len(resid_wt)
    F_wd    = (r_wd**2 / (1 - r_wd**2)) * (n_wd - 2) if r_wd**2 < 1 else 0
    p_wd    = 1 - stats.f.cdf(F_wd, 1, n_wd - 2)
    autocorr_detected = p_wd < 0.05

    use_cluster_se = hetero_detected or autocorr_detected
    cov_type_final = "clustered" if use_cluster_se else "unadjusted"

    # ── Fit ulang model terpilih dengan SE yang tepat (sesuai 4A.4) ────
    try:
        if best_name == "Fixed Effect Model (FEM)":
            res_final = PanelOLS.from_formula(
                f"{DEP_VAR} ~ 1 + {fv} + EntityEffects", data=df_panel
            ).fit(cov_type=cov_type_final)
        elif best_name == "Random Effect Model (REM)":
            res_final = RandomEffects.from_formula(
                f"{DEP_VAR} ~ 1 + {fv}", data=df_panel
            ).fit(cov_type=cov_type_final)
        else:
            res_final = PooledOLS.from_formula(
                f"{DEP_VAR} ~ 1 + {fv}", data=df_panel
            ).fit(cov_type=cov_type_final)
    except Exception:
        res_final = _res_for_diag  # fallback ke unadjusted

    # ── Juga fit ulang res_cem, res_fem, res_rem dengan cov yang sama
    # (untuk tab koefisien, agar konsisten dengan res_final)
    try:
        res_cem_final = PooledOLS.from_formula(
            f"{DEP_VAR} ~ 1 + {fv}", data=df_panel
        ).fit(cov_type=cov_type_final)
        res_fem_final = PanelOLS.from_formula(
            f"{DEP_VAR} ~ 1 + {fv} + EntityEffects", data=df_panel
        ).fit(cov_type=cov_type_final)
        res_rem_final = RandomEffects.from_formula(
            f"{DEP_VAR} ~ 1 + {fv}", data=df_panel
        ).fit(cov_type=cov_type_final)
    except Exception:
        res_cem_final, res_fem_final, res_rem_final = res_cem, res_fem, res_rem

    # ── R² Within / Between / Overall ──────────────────────────────────
    def get_r2_all(res):
        r2w = r2b = r2o = None
        try: r2w = res.rsquared_within
        except: pass
        try: r2b = res.rsquared_between
        except: pass
        try: r2o = res.rsquared_overall
        except: pass
        try:
            if not hasattr(res, 'rsquared_within'): r2w = res.rsquared
        except: pass
        return r2w, r2b, r2o

    r2w_final, r2b_final, r2o_final = get_r2_all(res_final)

    # ── Out-of-sample prediction (y_hat = Xβ + α_i untuk FEM) ──────────
    params      = res_final.params
    shared_vars = [v for v in INDEP_VARS if v in params.index]
    intercept   = params.get("Intercept", 0)

    if best_name == "Fixed Effect Model (FEM)":
        # Fitted train dengan estimated effects
        try:
            fitted_structural = res_final.fitted_values.iloc[:, 0]
            fitted_effects    = res_final.estimated_effects.iloc[:, 0]
            y_fitted_train    = (fitted_structural + fitted_effects).reindex(df_panel[DEP_VAR].index).values
        except Exception:
            X_train_np  = df_panel[shared_vars].values
            y_hat_tr    = intercept + X_train_np @ params[shared_vars].values
            alpha_i_fb  = pd.Series(
                df_panel[DEP_VAR].values - y_hat_tr,
                index=df_panel.index
            ).groupby(level="entity").mean()
            alpha_train = alpha_i_fb.reindex(df_panel.index.get_level_values("entity")).fillna(0).values
            y_fitted_train = y_hat_tr + alpha_train

        # α_i dari residual
        alpha_i = pd.Series(
            df_panel[DEP_VAR].values - y_fitted_train,
            index=df_panel.index
        ).groupby(level="entity").mean()
        # Lebih akurat: rata-rata residual dari model tanpa α_i
        X_train_np = df_panel[shared_vars].values
        y_hat_base = intercept + X_train_np @ params[shared_vars].values
        alpha_i    = pd.Series(
            df_panel[DEP_VAR].values - y_hat_base,
            index=df_panel.index
        ).groupby(level="entity").mean()
        # Final train prediction
        alpha_train_arr = alpha_i.reindex(df_panel.index.get_level_values("entity")).fillna(0).values
        y_fitted_train  = y_hat_base + alpha_train_arr

        # Test prediction
        df_test_r    = df_panel_test.reset_index()
        X_test_np    = df_test_r[shared_vars].values
        alpha_mapped = df_test_r["entity"].map(alpha_i).fillna(0).values
        y_pred_panel = intercept + X_test_np @ params[shared_vars].values + alpha_mapped
        y_test_panel = df_test_r[DEP_VAR].values
    else:
        y_fitted_train = intercept + df_panel[shared_vars].values @ params[shared_vars].values
        y_test_panel   = df_panel_test[DEP_VAR].values
        y_pred_panel   = intercept + df_panel_test[shared_vars].values @ params[shared_vars].values

    # Metrik train
    y_actual_train = df_panel[DEP_VAR].values
    panel_r2_train   = r2_score(y_actual_train, y_fitted_train)
    panel_rmse_train = np.sqrt(mean_squared_error(y_actual_train, y_fitted_train))
    panel_mae_train  = mean_absolute_error(y_actual_train, y_fitted_train)
    panel_mape_train = np.mean(np.abs((y_actual_train - y_fitted_train) / y_actual_train)) * 100

    # Metrik test
    panel_rmse = np.sqrt(mean_squared_error(y_test_panel, y_pred_panel))
    panel_mae  = mean_absolute_error(y_test_panel, y_pred_panel)
    panel_r2   = r2_score(y_test_panel, y_pred_panel)
    panel_mape = np.mean(np.abs((y_test_panel - y_pred_panel) / y_test_panel)) * 100

    panel_test_metrics  = {"R2": panel_r2,  "RMSE": panel_rmse,  "MAE": panel_mae,  "MAPE": panel_mape}
    panel_train_metrics = {"R2": panel_r2_train, "RMSE": panel_rmse_train, "MAE": panel_mae_train, "MAPE": panel_mape_train}

    # ── K-Means Clustering ───────────────────────────────────────────────
    df_mean = df_panel.reset_index().groupby("entity")[INDEP_VARS + [DEP_VAR]].mean()
    scaler    = StandardScaler()
    df_scaled = scaler.fit_transform(df_mean)

    sil_scores = {}
    for k in range(2, 8):
        km  = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        lbl = km.fit_predict(df_scaled)
        sil_scores[k] = silhouette_score(df_scaled, lbl)
    best_k = max(sil_scores, key=sil_scores.get)

    kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    df_mean["Cluster"] = kmeans_final.fit_predict(df_scaled)

    # Relabel: klaster 0 = IPM terendah
    cl_ipm_order = df_mean.groupby("Cluster")[DEP_VAR].mean().sort_values().index.tolist()
    relabel = {old: new for new, old in enumerate(cl_ipm_order)}
    df_mean["Cluster"] = df_mean["Cluster"].map(relabel)
    cluster_map = df_mean["Cluster"].to_dict()

    # Petakan ke df_panel
    df_panel_cl = df_panel.reset_index()
    df_panel_cl["Cluster"] = df_panel_cl["entity"].map(cluster_map)
    df_panel_cl = df_panel_cl.set_index(["entity", "time"])

    df_panel_test_cl = df_panel_test.reset_index()
    df_panel_test_cl["Cluster"] = df_panel_test_cl["entity"].map(cluster_map)
    df_panel_test_cl = df_panel_test_cl.set_index(["entity", "time"])

    # ── Regresi per klaster + evaluasi out-of-sample per klaster ────────
    cluster_results = {}
    cluster_eval_rows = []

    for c_id in range(best_k):
        df_sub      = df_panel_cl[df_panel_cl["Cluster"] == c_id]
        df_sub_test = df_panel_test_cl[df_panel_test_cl["Cluster"] == c_id]
        N_c = df_sub.index.get_level_values(0).nunique()

        if N_c < 5:
            cluster_results[c_id] = None
            continue
        try:
            rc_cem = PooledOLS.from_formula(f"{DEP_VAR} ~ 1 + {fv}", data=df_sub).fit(cov_type=cov_type_final)
            rc_fem = PanelOLS.from_formula(f"{DEP_VAR} ~ 1 + {fv} + EntityEffects", data=df_sub).fit(cov_type=cov_type_final)
            rc_rem = RandomEffects.from_formula(f"{DEP_VAR} ~ 1 + {fv}", data=df_sub).fit(cov_type=cov_type_final)

            pc = rc_fem.f_pooled.pval
            bfe2 = rc_fem.params; bre2 = rc_rem.params.reindex(bfe2.index).dropna()
            bfe2 = bfe2.reindex(bre2.index); diff2 = bfe2 - bre2
            cd2 = rc_fem.cov.loc[bre2.index, bre2.index] - rc_rem.cov.loc[bre2.index, bre2.index]
            try: H2 = max(float(diff2.values @ np.linalg.inv(cd2.values) @ diff2.values), 0)
            except: H2 = max(float(diff2.values @ np.linalg.pinv(cd2.values) @ diff2.values), 0)
            ph2 = 1 - stats.chi2.cdf(H2, len(bre2)) if H2 > 0 else 1.0

            if pc < 0.05 and ph2 < 0.05:
                best_cl_name, best_cl = "FEM", rc_fem
            elif pc < 0.05:
                best_cl_name, best_cl = "REM", rc_rem
            else:
                best_cl_name, best_cl = "CEM", rc_cem

            members = df_sub.index.get_level_values(0).unique().tolist()

            # Evaluasi per klaster (train)
            p_cl = best_cl.params
            iv_cl = [v for v in INDEP_VARS if v in p_cl.index]
            ic_cl = p_cl.get("Intercept", 0)

            if best_cl_name == "FEM":
                try:
                    fs_cl = best_cl.fitted_values.iloc[:, 0]
                    fe_cl = best_cl.estimated_effects.iloc[:, 0]
                    y_tr_cl = (fs_cl + fe_cl).reindex(df_sub[DEP_VAR].index).values
                except Exception:
                    Xtr = df_sub[iv_cl].values
                    yh  = ic_cl + Xtr @ p_cl[iv_cl].values
                    ai  = pd.Series(df_sub[DEP_VAR].values - yh, index=df_sub.index).groupby(level="entity").mean()
                    y_tr_cl = yh + ai.reindex(df_sub.index.get_level_values("entity")).fillna(0).values
            else:
                y_tr_cl = ic_cl + df_sub[iv_cl].values @ p_cl[iv_cl].values

            r2_train_cl   = r2_score(df_sub[DEP_VAR].values, y_tr_cl)
            rmse_train_cl = np.sqrt(mean_squared_error(df_sub[DEP_VAR].values, y_tr_cl))
            mae_train_cl  = mean_absolute_error(df_sub[DEP_VAR].values, y_tr_cl)

            # Evaluasi per klaster (test)
            r2_test_cl = rmse_test_cl = mae_test_cl = gap_r2_cl = np.nan
            if len(df_sub_test) > 0:
                try:
                    Xte = df_sub_test[iv_cl].values
                    if best_cl_name == "FEM":
                        Xtr2 = df_sub[iv_cl].values
                        yb2  = ic_cl + Xtr2 @ p_cl[iv_cl].values
                        ai2  = pd.Series(df_sub[DEP_VAR].values - yb2, index=df_sub.index).groupby(level="entity").mean()
                        al_te = df_sub_test.reset_index()["entity"].map(ai2).fillna(0).values
                        y_pred_cl = ic_cl + Xte @ p_cl[iv_cl].values + al_te
                    else:
                        y_pred_cl = ic_cl + Xte @ p_cl[iv_cl].values
                    y_act_cl   = df_sub_test[DEP_VAR].values
                    r2_test_cl   = r2_score(y_act_cl, y_pred_cl)
                    rmse_test_cl = np.sqrt(mean_squared_error(y_act_cl, y_pred_cl))
                    mae_test_cl  = mean_absolute_error(y_act_cl, y_pred_cl)
                    gap_r2_cl    = r2_train_cl - r2_test_cl
                except Exception:
                    pass

            cluster_results[c_id] = {
                "model": best_cl, "name": best_cl_name, "members": members,
                "p_chow": pc, "p_hausman": ph2,
                "r2_train": r2_train_cl, "rmse_train": rmse_train_cl, "mae_train": mae_train_cl,
                "r2_test":  r2_test_cl,  "rmse_test":  rmse_test_cl,  "mae_test":  mae_test_cl,
                "gap_r2": gap_r2_cl,
            }
            cluster_eval_rows.append({
                "Klaster"   : f"Klaster {c_id+1}",
                "Model"     : best_cl_name,
                "N Kab/Kota": N_c,
                "R² Train"  : round(r2_train_cl, 4),
                "R² Test"   : round(r2_test_cl, 4) if not np.isnan(r2_test_cl) else "–",
                "RMSE Train": round(rmse_train_cl, 4),
                "RMSE Test" : round(rmse_test_cl, 4) if not np.isnan(rmse_test_cl) else "–",
                "MAE Train" : round(mae_train_cl, 4),
                "MAE Test"  : round(mae_test_cl, 4) if not np.isnan(mae_test_cl) else "–",
                "Gap R²"    : round(gap_r2_cl, 4) if not np.isnan(gap_r2_cl) else "–",
            })
        except Exception as e:
            cluster_results[c_id] = None

    df_cluster_results = pd.DataFrame(cluster_eval_rows) if cluster_eval_rows else pd.DataFrame()

    # ── Random Forest (RF_VARS = INDEP_VARS) ────────────────────────────
    df_ml = df[["Kabupaten_Kota", "Tahun", DEP_VAR] + RF_VARS].dropna().copy()
    train_df = df_ml[df_ml["Tahun"].isin(TRAIN_YEARS)]
    test_df  = df_ml[df_ml["Tahun"].isin(TEST_YEARS)]

    X_train = train_df[RF_VARS].values
    y_train = train_df[DEP_VAR].values
    X_test  = test_df[RF_VARS].values
    y_test  = test_df[DEP_VAR].values

    param_grid = {
        "n_estimators"     : [200, 300],
        "max_depth"        : [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf" : [1, 2],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        param_grid, cv=tscv, scoring="neg_mean_squared_error",
        n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    rf_final   = gs.best_estimator_
    y_pred_rf  = rf_final.predict(X_test)
    y_pred_rf_train = rf_final.predict(X_train)

    rf_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_mae   = mean_absolute_error(y_test, y_pred_rf)
    rf_r2    = r2_score(y_test, y_pred_rf)
    rf_mape  = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

    rf_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
    rf_mae_train  = mean_absolute_error(y_train, y_pred_rf_train)
    rf_r2_train   = r2_score(y_train, y_pred_rf_train)
    rf_mape_train = np.mean(np.abs((y_train - y_pred_rf_train) / y_train)) * 100

    rf_test_metrics  = {"R2": rf_r2,       "RMSE": rf_rmse,       "MAE": rf_mae,       "MAPE": rf_mape}
    rf_train_metrics = {"R2": rf_r2_train, "RMSE": rf_rmse_train, "MAE": rf_mae_train, "MAPE": rf_mape_train}

    # GridSearch CV per fold untuk best params
    best_idx  = gs.best_index_
    cv_results = gs.cv_results_
    fold_mses = [abs(cv_results[f"split{i}_test_score"][best_idx]) for i in range(5)]
    fold_rmses = [m**0.5 for m in fold_mses]

    fi_df = pd.DataFrame({
        "Variabel"  : RF_VARS,
        "Importance": rf_final.feature_importances_,
        "Label"     : [LABEL_MAP.get(v, v) for v in RF_VARS],
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    mean_fi = fi_df["Importance"].mean()

    # ── Sintesis 2 Model (Tahap 5A sesuai notebook) ──────────────────────
    # Panel signifikansi
    sig_raw  = res_final.params.drop("Intercept", errors="ignore")
    pval_raw = res_final.pvalues.drop("Intercept", errors="ignore")
    syn_df = pd.DataFrame({
        "Koefisien": sig_raw.values,
        "p_value"  : pval_raw.values,
    }, index=sig_raw.index)
    syn_df["Signifikan"] = syn_df["p_value"] < 0.05
    syn_df["Arah"] = syn_df["Koefisien"].apply(lambda x: "Positif" if x > 0 else "Negatif")

    # RF feature importance
    fi_lookup = fi_df.set_index("Variabel")[["Importance"]]
    # Threshold koreksi (tanpa variabel dominan)
    fi_tanpa_dominan = fi_df[fi_df["Variabel"] != fi_df.iloc[0]["Variabel"]]["Importance"].mean()
    fi_lookup["Kategori_Koreksi"] = fi_lookup["Importance"].apply(
        lambda x: "Tinggi" if x >= fi_tanpa_dominan else "Rendah"
    )

    synthesis = syn_df.join(fi_lookup, how="outer")
    synthesis["Importance"]       = synthesis["Importance"].fillna(0)
    synthesis["Signifikan"]       = synthesis["Signifikan"].fillna(False)
    synthesis["Kategori_Koreksi"] = synthesis["Kategori_Koreksi"].fillna("Rendah")
    synthesis["Arah"]             = synthesis["Arah"].fillna("—")

    VARS_PERHATIAN_SYN = ["Ln_Jumlah_Sekolah", "Keluhan_Kesehatan"]
    synthesis["Perhatian"] = synthesis.index.map(
        lambda v: "⚠️ Perlu Perhatian" if v in VARS_PERHATIAN_SYN else "✅ Final"
    )

    def classify(row):
        sig = row["Signifikan"]
        fi  = row["Kategori_Koreksi"] == "Tinggi"
        if sig and fi:   return "VARIABEL KUNCI"
        elif sig:        return "VARIABEL STRUKTURAL"
        elif fi:         return "NON-LINEAR DRIVER"
        else:            return "VARIABEL MINOR"

    synthesis["Klasifikasi"] = synthesis.apply(classify, axis=1)
    synthesis["Konsistensi"] = synthesis.apply(
        lambda row: "Konsisten" if (row["Signifikan"] and row["Kategori_Koreksi"] == "Tinggi")
                    else ("Tidak Konsisten" if (row["Signifikan"] or row["Kategori_Koreksi"] == "Tinggi")
                          else "Lemah"), axis=1
    )
    synthesis["neg_log_p"] = synthesis["p_value"].apply(
        lambda p: -np.log10(max(float(p), 1e-10)) if pd.notna(p) and float(p) > 0 else 0.5
    )

    return {
        # Panel model objects (dengan SE final)
        "res_cem": res_cem_final, "res_fem": res_fem_final, "res_rem": res_rem_final,
        "res_final": res_final,
        "best_name": best_name, "best_res": res_final,
        "cov_type_final": cov_type_final,
        # Uji pemilihan
        "f_chow": f_chow, "p_chow": p_chow,
        "H_hausman": H, "p_hausman": p_hausman,
        "LM": LM, "p_lm": p_lm,
        # Uji asumsi klasik
        "jb_stat": jb_stat, "jb_pval": jb_pval,
        "sw_stat": sw_stat, "sw_pval": sw_pval,
        "lev_stat": lev_stat, "lev_pval": lev_pval,
        "hetero_detected": hetero_detected,
        "r_wd": r_wd, "F_wd": F_wd, "p_wd": p_wd,
        "autocorr_detected": autocorr_detected,
        "normality_ok": normality_ok,
        "use_cluster_se": use_cluster_se,
        "resid_arr": resid_arr,
        # R² lengkap
        "r2w_final": r2w_final, "r2b_final": r2b_final, "r2o_final": r2o_final,
        # Panel eval
        "panel_rmse": panel_rmse, "panel_mae": panel_mae, "panel_r2": panel_r2, "panel_mape": panel_mape,
        "panel_rmse_train": panel_rmse_train, "panel_mae_train": panel_mae_train,
        "panel_r2_train": panel_r2_train, "panel_mape_train": panel_mape_train,
        "panel_test_metrics": panel_test_metrics, "panel_train_metrics": panel_train_metrics,
        "y_test_panel": y_test_panel, "y_pred_panel": y_pred_panel,
        "df_panel_test": df_panel_test,
        # Clustering
        "sil_scores": sil_scores, "best_k": best_k,
        "cluster_results": cluster_results, "cluster_map": cluster_map,
        "df_mean": df_mean, "df_cluster_results": df_cluster_results,
        # RF
        "rf_final": rf_final, "best_params": gs.best_params_,
        "rf_rmse": rf_rmse, "rf_mae": rf_mae, "rf_r2": rf_r2, "rf_mape": rf_mape,
        "rf_rmse_train": rf_rmse_train, "rf_mae_train": rf_mae_train,
        "rf_r2_train": rf_r2_train, "rf_mape_train": rf_mape_train,
        "rf_test_metrics": rf_test_metrics, "rf_train_metrics": rf_train_metrics,
        "fold_mses": fold_mses, "fold_rmses": fold_rmses,
        "y_test": y_test, "y_pred_rf": y_pred_rf,
        "y_train": y_train, "y_pred_rf_train": y_pred_rf_train,
        "fi_df": fi_df, "mean_fi": mean_fi,
        "fi_tanpa_dominan": fi_tanpa_dominan,
        "test_df": test_df,
        # Sintesis
        "synthesis": synthesis,
    }


# LOAD DATA
df      = load_data()
geojson = load_geojson()
YEARS   = sorted(df["Tahun"].unique())
KAB_LIST = sorted(df["Kabupaten_Kota"].unique())

# SIDEBAR
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/"
        "Lambang_Provinsi_Jawa_Timur.svg/200px-Lambang_Provinsi_Jawa_Timur.svg.png",
        width=80
    )
    st.markdown("### 📊 Dashboard IPM Jawa Timur")
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["🏠 Overview", "🔍 Eksplorasi Data", "📐 Regresi Data Panel",
         "🌲 Random Forest", "🔗 Sintesis 2 Model", "🗺️ Peta IPM"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Filter Global**")
    sel_years = st.multiselect("Tahun", YEARS, default=YEARS, key="g_year")
    if not sel_years:
        sel_years = YEARS

    st.markdown("---")
    st.caption("Daniel Christopher Juwono · C14220006\nUniversitas Kristen Petra")

df_f = df[df["Tahun"].isin(sel_years)]

# HEADER
st.markdown("""
<div class="main-header">
  <h1>📊 Analisis Faktor-Faktor yang Mempengaruhi IPM di Jawa Timur</h1>
  <p>Metode: Regresi Data Panel (CEM / FEM / REM) + K-Means Clustering + Random Forest &nbsp;|&nbsp;
     38 Kabupaten/Kota · 2017–2024 · Train 2017–2022 · Test 2023–2024</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    ipm_mean = df_f["IPM"].mean()
    ipm_max  = df_f["IPM"].max()
    ipm_min  = df_f["IPM"].min()
    kab_max  = df_f.loc[df_f["IPM"].idxmax(), "Kabupaten_Kota"]
    kab_min  = df_f.loc[df_f["IPM"].idxmin(), "Kabupaten_Kota"]

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, sub in [
        (c1, f"{ipm_mean:.2f}", "Rata-rata IPM", f"{len(sel_years)} tahun terpilih"),
        (c2, f"{ipm_max:.2f}", "IPM Tertinggi", kab_max),
        (c3, f"{ipm_min:.2f}", "IPM Terendah",  kab_min),
        (c4, f"{len(df_f):,}", "Total Observasi", f"38 kab/kota × {len(sel_years)} tahun"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Tren IPM
    st.markdown('<div class="section-header">📈 Tren IPM Rata-Rata Jawa Timur</div>', unsafe_allow_html=True)
    tren = df.groupby("Tahun")["IPM"].agg(["mean", "min", "max"]).reset_index()
    tren.columns = ["Tahun", "Rata-rata", "Min", "Max"]

    fig_tren = go.Figure()
    fig_tren.add_trace(go.Scatter(
        x=tren["Tahun"], y=tren["Max"], name="Max",
        line=dict(width=0), showlegend=False
    ))
    fig_tren.add_trace(go.Scatter(
        x=tren["Tahun"], y=tren["Min"], name="Rentang Min–Max",
        fill="tonexty", fillcolor="rgba(46,109,164,0.12)",
        line=dict(width=0)
    ))
    fig_tren.add_trace(go.Scatter(
        x=tren["Tahun"], y=tren["Rata-rata"],
        name="Rata-rata Jawa Timur",
        mode="lines+markers+text",
        line=dict(color="#2e6da4", width=2.5),
        marker=dict(size=7),
        text=tren["Rata-rata"].round(2),
        textposition="top center", textfont=dict(size=10)
    ))
    fig_tren.update_layout(
        xaxis=dict(tickmode="linear"), yaxis_title="IPM",
        legend=dict(x=0.01, y=0.99),
        height=360, margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig_tren, use_container_width=True)

    # Top & Bottom 10
    st.markdown('<div class="section-header">🏙️ 10 Kabupaten/Kota IPM Tertinggi & Terendah</div>',
                unsafe_allow_html=True)
    mean_ipm = df_f.groupby("Kabupaten_Kota")["IPM"].mean().reset_index().sort_values("IPM")
    top_bot  = pd.concat([mean_ipm.head(10), mean_ipm.tail(10)])
    colors_tb = ["#e05252"] * 10 + ["#2e6da4"] * 10

    fig_bar = go.Figure(go.Bar(
        x=top_bot["IPM"], y=top_bot["Kabupaten_Kota"],
        orientation="h", marker_color=colors_tb,
        text=top_bot["IPM"].round(2), textposition="outside"
    ))
    fig_bar.update_layout(height=560, margin=dict(t=20, b=40, l=200))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Tren per kab/kota
    st.markdown('<div class="section-header">📊 Tren IPM per Kabupaten/Kota</div>', unsafe_allow_html=True)
    sel_kab = st.multiselect("Pilih Kabupaten/Kota:", KAB_LIST, default=KAB_LIST[:5], key="ov_kab")
    if sel_kab:
        fig_line = px.line(df[df["Kabupaten_Kota"].isin(sel_kab)],
                           x="Tahun", y="IPM", color="Kabupaten_Kota",
                           markers=True, color_discrete_sequence=PALETTE)
        fig_line.update_layout(height=360, margin=dict(t=20, b=40),
                                xaxis=dict(tickmode="linear"))
        st.plotly_chart(fig_line, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════
elif page == "🔍 Eksplorasi Data":
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "📊 Distribusi", "🔗 Korelasi", "📈 Scatter"])

    with tab1:
        cols_show = ["Tahun", "Kabupaten_Kota", "IPM",
                     "PDRB_Per_Orang", "Tingkat_Kemiskinan", "Tingkat_Pengangguran",
                     "Angka_Melek_Huruf", "Angka_Partisipasi_Murni",
                     "Sanitasi_Kesehatan", "Keluhan_Kesehatan"]
        st.dataframe(df_f[cols_show].reset_index(drop=True), use_container_width=True, height=440)
        st.download_button("⬇️ Unduh CSV", df_f[cols_show].to_csv(index=False).encode(),
                           "dataset_ipm.csv", "text/csv")

    with tab2:
        sel_var = st.selectbox("Variabel:", [DEP_VAR] + INDEP_VARS,
                               format_func=lambda v: LABEL_MAP.get(v, v))
        col_a, col_b = st.columns(2)
        with col_a:
            fig_h = px.histogram(df_f, x=sel_var, nbins=25,
                                 color_discrete_sequence=["#2e6da4"],
                                 title=f"Histogram — {LABEL_MAP.get(sel_var,sel_var)}")
            fig_h.update_layout(height=320, margin=dict(t=40,b=40))
            st.plotly_chart(fig_h, use_container_width=True)
        with col_b:
            fig_b = px.box(df_f, y=sel_var, x="Tahun",
                           color_discrete_sequence=["#2e6da4"],
                           title=f"Box Plot — {LABEL_MAP.get(sel_var,sel_var)}")
            fig_b.update_layout(height=320, margin=dict(t=40,b=40))
            st.plotly_chart(fig_b, use_container_width=True)

    with tab3:
        corr_vars = [DEP_VAR] + INDEP_VARS
        corr_mat  = df_f[corr_vars].corr()
        labels    = [LABEL_MAP.get(v, v) for v in corr_vars]

        fig_heat = px.imshow(
            corr_mat.values, x=labels, y=labels,
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=".2f", aspect="auto"
        )
        fig_heat.update_layout(height=520, margin=dict(t=20, b=80, l=120, r=20))
        st.plotly_chart(fig_heat, use_container_width=True)

        corr_ipm   = corr_mat[DEP_VAR].drop(DEP_VAR).sort_values()
        colors_cor = ["#e05252" if v < 0 else "#2e6da4" for v in corr_ipm]
        fig_cor = go.Figure(go.Bar(
            x=corr_ipm.values,
            y=[LABEL_MAP.get(v,v) for v in corr_ipm.index],
            orientation="h", marker_color=colors_cor,
            text=[f"{v:.3f}" for v in corr_ipm.values], textposition="outside"
        ))
        fig_cor.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_cor.update_layout(
            title="Korelasi Pearson terhadap IPM",
            xaxis=dict(range=[-1.1, 1.1]),
            height=380, margin=dict(t=40, b=40, l=200)
        )
        st.plotly_chart(fig_cor, use_container_width=True)

    with tab4:
        sel_x = st.selectbox("Variabel X:", INDEP_VARS,
                              format_func=lambda v: LABEL_MAP.get(v, v))
        fig_sc = px.scatter(
            df_f, x=sel_x, y="IPM", color="Tahun",
            hover_data=["Kabupaten_Kota"],
            color_continuous_scale="Blues",
            trendline="ols",
            labels={sel_x: LABEL_MAP.get(sel_x, sel_x)}
        )
        fig_sc.update_layout(height=440, margin=dict(t=20, b=40))
        st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE: REGRESI DATA PANEL
# ═══════════════════════════════════════════════════════════
elif page == "📐 Regresi Data Panel":
    with st.spinner("⏳ Menjalankan model regresi panel & K-Means clustering..."):
        M = run_models(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Pemilihan Model (4A)",
        "🔬 Uji Asumsi Klasik",
        "📋 Koefisien Model",
        "🗂️ Per Klaster (4B)",
        "📉 Evaluasi Out-of-Sample"
    ])

    # ── Tab 1: Pemilihan Model ──────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Hasil Tiga Uji Pemilihan Model</div>',
                    unsafe_allow_html=True)
        st.caption("Data training: 2017–2022 | Data testing: 2023–2024")

        def badge(cond, yes, no):
            if cond:
                return f'<span style="background:#28a745;color:white;padding:3px 9px;border-radius:4px;font-size:0.8rem">✅ {yes}</span>'
            return f'<span style="background:#dc3545;color:white;padding:3px 9px;border-radius:4px;font-size:0.8rem">❌ {no}</span>'

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🔵 Uji Chow** — CEM vs FEM")
            st.metric("F-statistic", f"{M['f_chow']:.4f}")
            st.metric("p-value",     f"{M['p_chow']:.4f}")
            st.markdown(badge(M['p_chow'] < 0.05, "FEM > CEM", "Gunakan CEM"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown("**🔵 Uji Hausman** — FEM vs REM")
            st.metric("H-statistic", f"{M['H_hausman']:.4f}")
            st.metric("p-value",     f"{M['p_hausman']:.4f}")
            st.markdown(badge(M['p_hausman'] < 0.05, "FEM konsisten", "Gunakan REM"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown("**🔵 Uji BP-LM** — CEM vs REM")
            st.metric("LM-statistic", f"{M['LM']:.4f}")
            st.metric("p-value",      f"{M['p_lm']:.4f}")
            st.markdown(badge(M['p_lm'] < 0.05, "REM > CEM", "Gunakan CEM"),
                        unsafe_allow_html=True)

        st.markdown("")
        se_label = "Cluster SE" if M["use_cluster_se"] else "SE Konvensional"
        st.success(f"⭐ **Model Terpilih: {M['best_name']}** | Standard Error: {se_label}")

        # Alur keputusan
        st.markdown('<div class="section-header">Alur Keputusan Pemilihan Model</div>',
                    unsafe_allow_html=True)
        col_left, _ = st.columns([2, 1])
        with col_left:
            alur_data = {
                "Uji": ["Uji Chow (CEM vs FEM)", "Uji Hausman (FEM vs REM)", "Uji BP-LM (CEM vs REM)"],
                "H0": ["Tidak ada fixed effect", "REM lebih efisien", "Varians individu = 0"],
                "Statistik": [f"F = {M['f_chow']:.4f}", f"H = {M['H_hausman']:.4f}", f"LM = {M['LM']:.4f}"],
                "p-value": [f"{M['p_chow']:.4f}", f"{M['p_hausman']:.4f}", f"{M['p_lm']:.4f}"],
                "Keputusan": [
                    "Tolak H0 → FEM" if M['p_chow'] < 0.05 else "Gagal Tolak → CEM",
                    "Tolak H0 → FEM" if M['p_hausman'] < 0.05 else "Gagal Tolak → REM",
                    "Tolak H0 → REM" if M['p_lm'] < 0.05 else "Gagal Tolak → CEM",
                ]
            }
            st.dataframe(pd.DataFrame(alur_data), use_container_width=True, hide_index=True)

    # ── Tab 2: Uji Asumsi Klasik ────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Uji Asumsi Klasik — Model Terpilih</div>',
                    unsafe_allow_html=True)
        st.caption(f"Model: {M['best_name']} | Data Training (2017–2022)")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**[1] Normalitas Residual**")
            st.caption("Jarque-Bera Test — H0: Residual berdistribusi normal")
            st.metric("JB Statistic", f"{M['jb_stat']:.4f}")
            st.metric("p-value", f"{M['jb_pval']:.4f}")
            if M["normality_ok"]:
                st.success("✅ Normal")
            else:
                st.warning("⚠️ Non-normal (n besar = wajar)")
            st.caption(f"Shapiro-Wilk: stat={M['sw_stat']:.4f}  p={M['sw_pval']:.4f}")

        with c2:
            st.markdown("**[2] Heteroskedastisitas**")
            st.caption("Levene Test per Entitas — H0: Varians homogen")
            st.metric("Levene Statistic", f"{M['lev_stat']:.4f}")
            st.metric("p-value", f"{M['lev_pval']:.4f}")
            if M["hetero_detected"]:
                st.warning("⚠️ Heteroskedastis → Cluster SE")
            else:
                st.success("✅ Homoskedastis")

        with c3:
            st.markdown("**[3] Autokorelasi**")
            st.caption("Wooldridge AR(1) Test — H0: Tidak ada AR(1)")
            st.metric("r(FD, lag-1)", f"{M['r_wd']:.4f}")
            st.metric("p-value", f"{M['p_wd']:.4f}")
            if M["autocorr_detected"]:
                st.warning("⚠️ AR(1) terdeteksi → Cluster SE")
            else:
                st.success("✅ Tidak ada AR(1)")

        st.markdown("")
        se_final = "**Cluster SE**" if M["use_cluster_se"] else "**SE Konvensional**"
        st.info(f"🔧 **Rekomendasi SE:** {se_final} (digunakan pada estimasi koefisien & evaluasi)")

        # R² Lengkap (Within / Between / Overall)
        st.markdown('<div class="section-header">R² Lengkap — Within / Between / Overall</div>',
                    unsafe_allow_html=True)
        r2_rows = []
        if M["r2w_final"] is not None:
            note = "← UTAMA untuk FEM" if "FEM" in M["best_name"] else ""
            r2_rows.append({"Jenis R²": "R² Within",  "Nilai": round(M["r2w_final"], 4), "Interpretasi": f"Variasi dalam entitas lintas waktu  {note}"})
        if M["r2b_final"] is not None:
            r2_rows.append({"Jenis R²": "R² Between", "Nilai": round(M["r2b_final"], 4), "Interpretasi": "Variasi antar entitas"})
        if M["r2o_final"] is not None:
            r2_rows.append({"Jenis R²": "R² Overall", "Nilai": round(M["r2o_final"], 4), "Interpretasi": "Variasi total (within + between)"})
        if r2_rows:
            st.dataframe(pd.DataFrame(r2_rows), use_container_width=False, hide_index=True)

        # Visualisasi residual
        st.markdown('<div class="section-header">Visualisasi Diagnostik Residual</div>',
                    unsafe_allow_html=True)
        resid_arr = M["resid_arr"]
        col_l, col_r = st.columns(2)
        with col_l:
            fig_hist = px.histogram(
                x=resid_arr, nbins=25, color_discrete_sequence=["#2e6da4"],
                title="Distribusi Residual", labels={"x": "Residual"}
            )
            fig_hist.update_layout(height=280, margin=dict(t=40, b=40))
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_r:
            # Q-Q Plot
            from scipy import stats as sp_stats
            (osm, osr), (slope, intercept_qq, r) = sp_stats.probplot(resid_arr)
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                         marker=dict(color="#2e6da4", size=4, opacity=0.6),
                                         name="Data"))
            fig_qq.add_trace(go.Scatter(x=[min(osm), max(osm)],
                                         y=[slope*min(osm)+intercept_qq, slope*max(osm)+intercept_qq],
                                         mode="lines", line=dict(color="red", dash="dash"),
                                         name="Garis Normal"))
            fig_qq.update_layout(title="Q-Q Plot Residual",
                                  xaxis_title="Theoretical Quantiles",
                                  yaxis_title="Sample Quantiles",
                                  height=280, margin=dict(t=40, b=40))
            st.plotly_chart(fig_qq, use_container_width=True)

    # ── Tab 3: Koefisien Model ──────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Tabel Koefisien Regresi</div>',
                    unsafe_allow_html=True)

        se_note = "Cluster SE" if M["use_cluster_se"] else "SE Konvensional"
        st.caption(f"Standard Error yang digunakan: {se_note}")

        sel_model = st.radio(
            "Tampilkan:",
            ["CEM (Pooled OLS)", "FEM (Fixed Effect)", "REM (Random Effect)"],
            horizontal=True
        )
        res_map = {
            "CEM (Pooled OLS)"   : M["res_cem"],
            "FEM (Fixed Effect)" : M["res_fem"],
            "REM (Random Effect)": M["res_rem"],
        }
        res = res_map[sel_model]

        # R² — gunakan atribut yang tersedia
        if sel_model == "FEM (Fixed Effect)":
            r2_val = res.rsquared_within
            r2_lbl = "R² Within"
        elif sel_model == "REM (Random Effect)":
            r2_val = res.rsquared_overall
            r2_lbl = "R² Overall"
        else:
            r2_val = res.rsquared
            r2_lbl = "R²"

        rows = []
        for v in res.params.index:
            p   = res.pvalues[v]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ci  = res.conf_int()
            ci_row = ci.iloc[list(res.params.index).index(v)]
            rows.append({
                "Variabel"      : LABEL_MAP.get(v, v),
                "Koefisien"     : round(res.params[v], 4),
                "Std. Error"    : round(res.std_errors[v], 4),
                "p-value"       : round(p, 4),
                "Sig"           : sig,
                "CI 95% Lower"  : round(ci_row.iloc[0], 4),
                "CI 95% Upper"  : round(ci_row.iloc[1], 4),
            })
        df_coef = pd.DataFrame(rows)

        def color_sig(val):
            return {
                "***": "color:#155724;font-weight:bold",
                "**" : "color:#1c7430;font-weight:bold",
                "*"  : "color:#856404;font-weight:bold",
                "ns" : "color:#721c24"
            }.get(val, "")

        st.dataframe(
            df_coef.style.applymap(color_sig, subset=["Sig"]),
            use_container_width=True, hide_index=True
        )
        st.markdown(f"**{r2_lbl} = {r2_val:.4f}** &nbsp;|&nbsp; "
                    f"\\*\\*\\* p<0.001 &nbsp; \\*\\* p<0.01 &nbsp; \\* p<0.05 &nbsp; ns = tidak signifikan")

        # Coefficient plot
        df_cp = df_coef[df_coef["Variabel"] != "Intercept"].copy()
        fig_cp = go.Figure()
        fig_cp.add_trace(go.Scatter(
            x=df_cp["Koefisien"], y=df_cp["Variabel"],
            mode="markers",
            marker=dict(
                color=["#2e6da4" if v > 0 else "#e05252" for v in df_cp["Koefisien"]],
                size=10
            ),
            error_x=dict(
                type="data", symmetric=False,
                array=(df_cp["CI 95% Upper"] - df_cp["Koefisien"]).tolist(),
                arrayminus=(df_cp["Koefisien"] - df_cp["CI 95% Lower"]).tolist(),
                color="gray"
            )
        ))
        fig_cp.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_cp.update_layout(
            title=f"Coefficient Plot — {sel_model} ({se_note})",
            xaxis_title="Koefisien (CI 95%)",
            height=400, margin=dict(t=40, b=40, l=220)
        )
        st.plotly_chart(fig_cp, use_container_width=True)

    # ── Tab 4: Per Klaster ──────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">K-Means Clustering + Regresi Panel per Klaster</div>',
                    unsafe_allow_html=True)
        st.info(f"Jumlah klaster optimal (Silhouette): **k = {M['best_k']}**")

        # Silhouette chart
        sil_df = pd.DataFrame({"k": list(M["sil_scores"].keys()),
                                "Silhouette Score": list(M["sil_scores"].values())})
        fig_sil = px.bar(sil_df, x="k", y="Silhouette Score",
                         color_discrete_sequence=["#2e6da4"],
                         title="Silhouette Score per k")
        fig_sil.add_vline(x=M["best_k"], line_dash="dash", line_color="red",
                          annotation_text=f"k terpilih = {M['best_k']}")
        fig_sil.update_layout(height=280, margin=dict(t=40, b=40))
        st.plotly_chart(fig_sil, use_container_width=True)

        # Rata-rata IPM per klaster
        dm = M["df_mean"].copy()
        ipm_cl = dm.groupby("Cluster")["IPM"].mean().reset_index()
        ipm_cl.columns = ["Klaster", "Rata-rata IPM"]
        ipm_cl["Klaster"] = ipm_cl["Klaster"].apply(lambda x: f"Klaster {x+1}")
        ipm_cl["Rata-rata IPM"] = ipm_cl["Rata-rata IPM"].round(3)

        fig_ipm_cl = px.bar(ipm_cl, x="Klaster", y="Rata-rata IPM",
                            color_discrete_sequence=["#2e6da4"],
                            title="Rata-rata IPM per Klaster",
                            text="Rata-rata IPM")
        fig_ipm_cl.update_layout(height=280, margin=dict(t=40, b=40))
        st.plotly_chart(fig_ipm_cl, use_container_width=True)

        # Tabel evaluasi per klaster (sesuai Tahap 5B notebook)
        if not M["df_cluster_results"].empty:
            st.markdown('<div class="section-header">Evaluasi Out-of-Sample per Klaster</div>',
                        unsafe_allow_html=True)
            st.dataframe(M["df_cluster_results"], use_container_width=True, hide_index=True)

            # Perbandingan Global vs Cluster terbaik
            best_cluster_rmse_val = M["df_cluster_results"]["RMSE Test"].replace("–", np.nan)
            best_cluster_rmse_val = pd.to_numeric(best_cluster_rmse_val, errors="coerce").min()
            if not np.isnan(best_cluster_rmse_val):
                if M["panel_rmse"] <= best_cluster_rmse_val:
                    st.success("📌 **Kesimpulan:** Model panel global lebih stabil dibanding pendekatan berbasis klaster.")
                else:
                    st.info("📌 **Kesimpulan:** Pendekatan berbasis klaster menunjukkan potensi peningkatan akurasi prediksi.")

        # Expander per klaster
        for c_id in range(M["best_k"]):
            cr = M["cluster_results"].get(c_id)
            members = dm[dm["Cluster"] == c_id].index.tolist() if cr is None else cr.get("members", [])
            with st.expander(f"🗂️ Klaster {c_id+1} — {len(members)} kabupaten/kota", expanded=(c_id == 0)):
                st.caption("Anggota: " + " · ".join(members))
                if cr is not None:
                    r2_cl_val = cr["model"].rsquared_within if cr["name"]=="FEM" else (
                        cr["model"].rsquared_overall if cr["name"]=="REM" else cr["model"].rsquared)
                    st.info(f"Model terpilih: **{cr['name']}** &nbsp;|&nbsp; R² = {r2_cl_val:.4f} &nbsp;|&nbsp; "
                            f"p_chow = {cr['p_chow']:.4f} &nbsp;|&nbsp; p_hausman = {cr['p_hausman']:.4f}")
                    rows_cl = []
                    for v in cr["model"].params.index:
                        p   = cr["model"].pvalues[v]
                        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                        rows_cl.append({
                            "Variabel"   : LABEL_MAP.get(v, v),
                            "Koefisien"  : round(cr["model"].params[v], 4),
                            "p-value"    : round(p, 4),
                            "Signifikansi": sig,
                        })
                    st.dataframe(
                        pd.DataFrame(rows_cl).style.applymap(color_sig, subset=["Signifikansi"]),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.warning("Observasi tidak cukup untuk estimasi model (N < 5).")

    # ── Tab 5: Evaluasi Out-of-Sample ──────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">Evaluasi Out-of-Sample — Data Test 2023–2024</div>',
                    unsafe_allow_html=True)
        st.caption(f"Model terbaik: {M['best_name']} | "
                   f"Prediksi menggunakan: y_hat = Xβ + α_i (entity fixed effects)")

        # Tabel perbandingan train vs test dengan MAPE
        gap_r2 = M['panel_r2_train'] - M['panel_r2']
        comp_df = pd.DataFrame({
            "Metrik" : ["R²", "RMSE", "MAE", "MAPE (%)"],
            "Train (in-sample) 2017–2022" : [
                f"{M['panel_r2_train']:.4f}",
                f"{M['panel_rmse_train']:.4f}",
                f"{M['panel_mae_train']:.4f}",
                f"{M['panel_mape_train']:.2f}%",
            ],
            "Test (out-of-sample) 2023–2024" : [
                f"{M['panel_r2']:.4f}",
                f"{M['panel_rmse']:.4f}",
                f"{M['panel_mae']:.4f}",
                f"{M['panel_mape']:.2f}%",
            ],
        })
        st.dataframe(comp_df, use_container_width=False, hide_index=True)
        if gap_r2 <= 0.05:
            st.success(f"✅ Gap R² (Train − Test) = {gap_r2:.4f} → Model stabil, tidak overfit")
        else:
            st.warning(f"⚠️ Gap R² (Train − Test) = {gap_r2:.4f} → Perlu dicek potensi overfit")

        # Metrik test saja (besar)
        ca, cb, cc, cd = st.columns(4)
        ca.metric("R² (Test)",    f"{M['panel_r2']:.4f}",   f"{M['panel_r2']-M['panel_r2_train']:.4f} vs train")
        cb.metric("RMSE (Test)",  f"{M['panel_rmse']:.4f}")
        cc.metric("MAE (Test)",   f"{M['panel_mae']:.4f}")
        cd.metric("MAPE (Test)",  f"{M['panel_mape']:.2f}%")

        # Plot aktual vs prediksi
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            y=M["y_test_panel"], name="Aktual",
            mode="lines+markers", line=dict(color="#2e6da4")
        ))
        fig_pred.add_trace(go.Scatter(
            y=M["y_pred_panel"], name="Prediksi",
            mode="lines+markers", line=dict(color="#e05252", dash="dash")
        ))
        fig_pred.update_layout(
            title=f"Aktual vs Prediksi — {M['best_name']} (Test 2023–2024)",
            xaxis_title="Observasi Test", yaxis_title="IPM",
            height=380, margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Scatter aktual vs prediksi
        mn = min(M["y_test_panel"].min(), M["y_pred_panel"].min()) - 0.5
        mx = max(M["y_test_panel"].max(), M["y_pred_panel"].max()) + 0.5
        fig_sc2 = go.Figure()
        fig_sc2.add_trace(go.Scatter(
            x=M["y_test_panel"], y=M["y_pred_panel"],
            mode="markers", marker=dict(color="#2e6da4", size=7, opacity=0.7),
            name="Prediksi vs Aktual"
        ))
        fig_sc2.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color="red", dash="dash"), name="Garis Sempurna"
        ))
        fig_sc2.update_layout(
            xaxis_title="IPM Aktual", yaxis_title="IPM Prediksi",
            title="Scatter Aktual vs Prediksi (Test)",
            height=360, margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig_sc2, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE: RANDOM FOREST
# ═══════════════════════════════════════════════════════════
elif page == "🌲 Random Forest":
    with st.spinner("⏳ Melatih Random Forest (GridSearch + TimeSeriesSplit 5-fold)..."):
        M = run_models(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Metrik & Hyperparameter", "📊 Feature Importance", "📉 Aktual vs Prediksi", "🔍 SHAP", "📐 PDP"])

    with tab1:
        st.markdown('<div class="section-header">Metrik Evaluasi — Perbandingan Train vs Test</div>',
                    unsafe_allow_html=True)
        # Tabel train vs test dengan MAPE (sesuai Tahap 4C.2 notebook)
        rf_comp = pd.DataFrame({
            "Metrik"  : ["R²", "RMSE", "MAE", "MAPE (%)"],
            "Train (2017–2022)": [
                f"{M['rf_r2_train']:.4f}",
                f"{M['rf_rmse_train']:.4f}",
                f"{M['rf_mae_train']:.4f}",
                f"{M['rf_mape_train']:.2f}%",
            ],
            "Test (2023–2024)": [
                f"{M['rf_r2']:.4f}",
                f"{M['rf_rmse']:.4f}",
                f"{M['rf_mae']:.4f}",
                f"{M['rf_mape']:.2f}%",
            ],
        })
        st.dataframe(rf_comp, use_container_width=False, hide_index=True)
        gap_rf = M["rf_r2_train"] - M["rf_r2"]
        if gap_rf <= 0.15:
            st.success(f"✅ Gap R² = {gap_rf:.4f} → Tidak overfitting")
        else:
            st.warning(f"⚠️ Gap R² = {gap_rf:.4f} → Potensi overfitting")

        st.markdown('<div class="section-header">Hyperparameter Terbaik (GridSearchCV)</div>',
                    unsafe_allow_html=True)
        bp = M["best_params"]
        st.dataframe(
            pd.DataFrame({"Parameter": list(bp.keys()), "Nilai": [str(v) for v in bp.values()]}),
            use_container_width=False, hide_index=True
        )

        # CV scores per fold (sesuai Tahap 4C.1 notebook)
        st.markdown('<div class="section-header">CV Scores per Fold — Best Parameter</div>',
                    unsafe_allow_html=True)
        fold_df = pd.DataFrame({
            "Fold"     : [f"Fold {i+1}" for i in range(5)],
            "MSE"      : [round(m, 4) for m in M["fold_mses"]],
            "RMSE"     : [round(r, 4) for r in M["fold_rmses"]],
        })
        fold_df.loc[len(fold_df)] = ["Mean", round(np.mean(M["fold_mses"]), 4), round(np.mean(M["fold_rmses"]), 4)]
        fold_df.loc[len(fold_df)] = ["Std",  round(np.std(M["fold_mses"]), 4),  "–"]
        st.dataframe(fold_df, use_container_width=False, hide_index=True)

        # Bar chart MSE per fold
        fig_fold = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=M["fold_mses"],
            marker_color="#2166ac",
            text=[f"{v:.4f}" for v in M["fold_mses"]], textposition="outside"
        ))
        fig_fold.add_hline(y=np.mean(M["fold_mses"]), line_dash="dash", line_color="red",
                           annotation_text=f"Mean = {np.mean(M['fold_mses']):.4f}")
        fig_fold.update_layout(title="CV MSE per Fold — Best Parameters",
                                yaxis_title="MSE", height=280, margin=dict(t=40, b=40))
        st.plotly_chart(fig_fold, use_container_width=True)

        st.caption("Variabel RF = INDEP_VARS (10 variabel, sama dengan regresi panel) | random_state=42 | TimeSeriesSplit n=5")

        # Perbandingan metrik panel vs RF
        st.markdown('<div class="section-header">Perbandingan Panel vs Random Forest (Test Set)</div>',
                    unsafe_allow_html=True)
        comp = pd.DataFrame({
            "Model"  : [M["best_name"], "Random Forest"],
            "R²"     : [round(M["panel_r2"],   4), round(M["rf_r2"],   4)],
            "RMSE"   : [round(M["panel_rmse"], 4), round(M["rf_rmse"], 4)],
            "MAE"    : [round(M["panel_mae"],  4), round(M["rf_mae"],  4)],
            "MAPE (%)": [round(M["panel_mape"], 2), round(M["rf_mape"], 2)],
        })
        st.dataframe(comp, use_container_width=False, hide_index=True)

    with tab2:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)

        fi = M["fi_df"].copy()
        fi["Kategori"] = fi["Importance"].apply(lambda x: "Tinggi" if x >= M["mean_fi"] else "Rendah")
        fi["Keterangan"] = fi["Variabel"].apply(
            lambda v: "⚠️ Perlu Perhatian" if v in FI_PERHATIAN else "✅ Final"
        )
        bar_colors = ["#f4a261" if v in FI_PERHATIAN else ("#2166ac" if k == "Tinggi" else "#92c5de")
                      for v, k in zip(fi["Variabel"], fi["Kategori"])]

        fig_fi = go.Figure(go.Bar(
            x=fi["Importance"],
            y=fi["Label"],
            orientation="h",
            marker_color=bar_colors,
            text=fi["Importance"].round(4), textposition="outside"
        ))
        fig_fi.add_vline(x=M["mean_fi"], line_dash="dash", line_color="red",
                         annotation_text=f"Rata-rata = {M['mean_fi']:.3f}")
        fig_fi.update_layout(
            title="Feature Importance — Random Forest (MDI)",
            xaxis_title="Importance Score",
            height=440, margin=dict(t=40, b=40, l=230)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        fi_show = fi[["Label", "Importance", "Kategori", "Keterangan"]].copy()
        fi_show.columns = ["Variabel", "Importance Score", "Kategori", "Keterangan"]
        fi_show["Importance Score"] = fi_show["Importance Score"].round(4)
        fi_show.index = range(1, len(fi_show)+1)
        st.dataframe(fi_show, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Aktual vs Prediksi — Data Test 2023–2024</div>',
                    unsafe_allow_html=True)

        test_df_out = M["test_df"][["Kabupaten_Kota", "Tahun", DEP_VAR]].copy()
        test_df_out["Prediksi_RF"] = M["y_pred_rf"]
        test_df_out["Residual"]    = (test_df_out["IPM"] - test_df_out["Prediksi_RF"]).round(4)

        # Scatter
        mn = test_df_out["IPM"].min() - 1
        mx = test_df_out["IPM"].max() + 1
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=test_df_out["IPM"], y=test_df_out["Prediksi_RF"],
            mode="markers",
            marker=dict(color="#2e6da4", size=7, opacity=0.75),
            text=test_df_out["Kabupaten_Kota"],
            name="Prediksi"
        ))
        fig_sc.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color="red", dash="dash"), name="Garis Sempurna"
        ))
        fig_sc.update_layout(
            xaxis_title="IPM Aktual", yaxis_title="IPM Prediksi",
            title="Scatter Aktual vs Prediksi RF (2023–2024)",
            height=420, margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.dataframe(test_df_out.round(3), use_container_width=True, hide_index=True)

    with tab4:
        st.markdown('<div class="section-header">SHAP Summary Plot — Train vs Test</div>',
                    unsafe_allow_html=True)
        try:
            import shap
            import matplotlib.pyplot as plt

            explainer         = shap.TreeExplainer(M["rf_final"])
            shap_values_train = explainer.shap_values(M["test_df"][RF_VARS].values)  # pakai test agar lebih cepat
            shap_values_test  = explainer.shap_values(M["test_df"][RF_VARS].values)

            fig_shap, axes_shap = plt.subplots(1, 2, figsize=(16, 7))
            plt.sca(axes_shap[0])
            shap.summary_plot(shap_values_train,
                              M["test_df"][RF_VARS].values,
                              feature_names=RF_VARS, show=False, plot_size=None)
            axes_shap[0].set_title("SHAP — Train (representatif)", fontsize=10, fontweight="bold")

            plt.sca(axes_shap[1])
            shap.summary_plot(shap_values_test,
                              M["test_df"][RF_VARS].values,
                              feature_names=RF_VARS, show=False, plot_size=None)
            axes_shap[1].set_title("SHAP — Test (2023–2024)", fontsize=10, fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)
        except ImportError:
            st.error("Library `shap` tidak terinstall. Jalankan: pip install shap")
        except Exception as e:
            st.error(f"SHAP error: {e}")

    with tab5:
        st.markdown('<div class="section-header">Partial Dependence Plots — Semua Variabel</div>',
                    unsafe_allow_html=True)
        try:
            from sklearn.inspection import PartialDependenceDisplay
            import matplotlib.pyplot as plt

            n_cols_pdp = 2
            n_rows_pdp = math.ceil(len(RF_VARS) / n_cols_pdp)
            fig_pdp, axes_pdp = plt.subplots(n_rows_pdp, n_cols_pdp,
                                              figsize=(14, n_rows_pdp * 4))
            axes_pdp = axes_pdp.flatten()
            X_train_pdp = M["test_df"][RF_VARS].values  # gunakan test sebagai referensi

            VARS_PERHATIAN_PDP = ["Ln_Jumlah_Sekolah", "Jumlah_Penduduk", "Keluhan_Kesehatan"]
            for i, (var, ax) in enumerate(zip(RF_VARS, axes_pdp)):
                feat_idx = RF_VARS.index(var)
                PartialDependenceDisplay.from_estimator(
                    M["rf_final"], X_train_pdp, features=[feat_idx],
                    feature_names=RF_VARS, ax=ax,
                    line_kw={"color": "#F4A261" if var in VARS_PERHATIAN_PDP else "#2166ac", "lw": 2.5}
                )
                label = "⚠️ " if var in VARS_PERHATIAN_PDP else ""
                ax.set_title(f"PDP: {label}{LABEL_MAP.get(var, var)}", fontsize=9, fontweight="bold")
                ax.set_xlabel(LABEL_MAP.get(var, var), fontsize=8)
                ax.grid(True, ls=":", alpha=0.5)

            for j in range(i + 1, len(axes_pdp)):
                fig_pdp.delaxes(axes_pdp[j])

            plt.suptitle("Partial Dependence Plots — Semua Variabel Final\n(Oranye = variabel perlu perhatian)",
                         fontsize=12, fontweight="bold", y=1.01)
            plt.tight_layout(pad=2.5)
            st.pyplot(fig_pdp)
            plt.close(fig_pdp)
        except Exception as e:
            st.error(f"PDP error: {e}")




# ═══════════════════════════════════════════════════════════
# PAGE: SINTESIS 2 MODEL (Tahap 5)
# ═══════════════════════════════════════════════════════════
elif page == "🔗 Sintesis 2 Model":
    with st.spinner("⏳ Menjalankan sintesis panel & Random Forest..."):
        M = run_models(df)

    st.markdown('<div class="section-header">Tahap 5A — Matriks Sintesis: Panel × Random Forest</div>',
                unsafe_allow_html=True)
    st.caption(
        "Setiap variabel diklasifikasikan berdasarkan signifikansi statistik (regresi panel) "
        "dan kontribusi prediktif (feature importance RF)."
    )

    synthesis = M["synthesis"].copy()

    # ── Tabel klasifikasi ───────────────────────────────────────────────
    COLOR_KLS = {
        "VARIABEL KUNCI":      "#2ca25f",
        "VARIABEL STRUKTURAL": "#2b8cbe",
        "NON-LINEAR DRIVER":   "#f03b20",
        "VARIABEL MINOR":      "#bdbdbd",
    }
    ICON_KLS = {
        "VARIABEL KUNCI":      "🟢",
        "VARIABEL STRUKTURAL": "🟡",
        "NON-LINEAR DRIVER":   "🔴",
        "VARIABEL MINOR":      "⚪",
    }

    syn_display = synthesis.reset_index().rename(columns={"index": "Variabel"})
    syn_display["Label"] = syn_display["Variabel"].apply(lambda v: LABEL_MAP.get(v, v))
    syn_display["Signifikan"] = syn_display["Signifikan"].map({True: "Ya", False: "Tidak"})
    syn_display["p-value"] = syn_display["p_value"].apply(
        lambda p: f"{p:.4f}" if pd.notna(p) else "—"
    )
    syn_display["FI Score"] = syn_display["Importance"].round(4)
    syn_display["Klasifikasi_Label"] = syn_display["Klasifikasi"].apply(
        lambda k: f"{ICON_KLS.get(k,'')} {k}"
    )
    cols_show_syn = ["Label", "Signifikan", "Arah", "p-value", "FI Score",
                     "Klasifikasi_Label"]
    st.dataframe(syn_display[cols_show_syn].rename(columns={"Label": "Variabel"}),
                 use_container_width=True, hide_index=True)

    # ── Scatter Matriks (FI vs -log10 p) ───────────────────────────────
    st.markdown('<div class="section-header">Visualisasi Matriks Sintesis</div>',
                unsafe_allow_html=True)
    st.caption("Sumbu X = Feature Importance RF | Sumbu Y = −log₁₀(p-value) panel | "
               "Kuadran kanan-atas = Variabel Kunci")

    fig_syn = go.Figure()
    VARS_PERHATIAN_SYN = ["Ln_Jumlah_Sekolah", "Keluhan_Kesehatan"]

    for _, row in synthesis.iterrows():
        color = COLOR_KLS.get(row["Klasifikasi"], "#999")
        edge  = "#F4A261" if row.name in VARS_PERHATIAN_SYN else "white"
        lw    = 2.5 if row.name in VARS_PERHATIAN_SYN else 1.5
        label = LABEL_MAP.get(row.name, row.name)
        fig_syn.add_trace(go.Scatter(
            x=[row["Importance"]], y=[row["neg_log_p"]],
            mode="markers+text",
            marker=dict(color=color, size=14, line=dict(color=edge, width=lw)),
            text=[label],
            textposition="top right",
            textfont=dict(size=9),
            name=row["Klasifikasi"],
            showlegend=False,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"FI = {row['Importance']:.4f}<br>"
                f"p-value = {row.get('p_value', 'n/a')}<br>"
                f"Klasifikasi: {row['Klasifikasi']}"
            )
        ))

    threshold_y = -np.log10(0.05)
    fig_syn.add_hline(y=threshold_y, line_dash="dash", line_color="gray",
                      annotation_text="p=0.05")
    fig_syn.add_vline(x=M["fi_tanpa_dominan"], line_dash="dot", line_color="gray",
                      annotation_text=f"FI threshold={M['fi_tanpa_dominan']:.3f}")

    # Legend manual
    for kls, clr in COLOR_KLS.items():
        fig_syn.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=clr, size=12),
            name=kls
        ))
    fig_syn.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="white", size=12, line=dict(color="#F4A261", width=2.5)),
        name="⚠️ border oranye = perlu perhatian"
    ))

    fig_syn.update_layout(
        xaxis_title="Feature Importance (Random Forest)",
        yaxis_title="−log₁₀(p-value) dari Regresi Panel",
        title=f"Matriks Sintesis: Signifikansi × Feature Importance<br>Model Panel: {M['best_name']}",
        height=520, margin=dict(t=60, b=40, l=60, r=40),
        legend=dict(x=1.01, y=1, xanchor="left")
    )
    st.plotly_chart(fig_syn, use_container_width=True)

    # ── Head-to-head Panel vs RF (sesuai Tahap 5A notebook) ─────────────
    st.markdown('<div class="section-header">Perbandingan Head-to-Head: Panel vs Random Forest (Test Set)</div>',
                unsafe_allow_html=True)
    hth_rows = []
    for m_key in ["R2", "RMSE", "MAE", "MAPE"]:
        pv = M["panel_test_metrics"].get(m_key, float("nan"))
        rv = M["rf_test_metrics"].get(m_key, float("nan"))
        winner = "Panel" if (pv >= rv if m_key == "R2" else pv <= rv) else "RF"
        hth_rows.append({
            "Metrik"         : m_key if m_key != "MAPE" else "MAPE (%)",
            "Panel (Global)" : round(pv, 4),
            "Random Forest"  : round(rv, 4),
            "Lebih Baik"     : f"→ {winner}",
        })
    st.dataframe(pd.DataFrame(hth_rows), use_container_width=False, hide_index=True)

    # ── Tahap 5B: Perbandingan Global vs Cluster ─────────────────────────
    st.markdown('<div class="section-header">Tahap 5B — Perbandingan Global vs Klaster</div>',
                unsafe_allow_html=True)
    if not M["df_cluster_results"].empty:
        st.dataframe(M["df_cluster_results"], use_container_width=True, hide_index=True)
        best_cl_rmse_s = M["df_cluster_results"]["RMSE Test"].replace("–", np.nan)
        best_cl_rmse_v = pd.to_numeric(best_cl_rmse_s, errors="coerce").min()
        if not np.isnan(best_cl_rmse_v):
            if M["panel_rmse"] <= best_cl_rmse_v:
                st.success("📌 Model panel global lebih stabil dibanding pendekatan berbasis klaster.")
            else:
                st.info("📌 Pendekatan berbasis klaster menunjukkan potensi peningkatan akurasi prediksi.")
    else:
        st.info("Data evaluasi klaster tidak tersedia.")

    # ── Tahap 5C: Kesimpulan Temuan Utama ───────────────────────────────
    st.markdown('<div class="section-header">Tahap 5C — Temuan Utama</div>',
                unsafe_allow_html=True)
    variabel_kunci   = synthesis[synthesis["Klasifikasi"] == "VARIABEL KUNCI"].index.tolist()
    variabel_str     = synthesis[synthesis["Klasifikasi"] == "VARIABEL STRUKTURAL"].index.tolist()
    non_linear_drv   = synthesis[synthesis["Klasifikasi"] == "NON-LINEAR DRIVER"].index.tolist()
    variabel_minor   = synthesis[synthesis["Klasifikasi"] == "VARIABEL MINOR"].index.tolist()

    col_l, col_r = st.columns(2)
    with col_l:
        if variabel_kunci:
            st.success(f"🟢 **Variabel Kunci** (signifikan + FI tinggi):\n" +
                       "\n".join([f"- {LABEL_MAP.get(v,v)}" for v in variabel_kunci]))
        if variabel_str:
            st.info(f"🟡 **Variabel Struktural** (signifikan, FI rendah):\n" +
                    "\n".join([f"- {LABEL_MAP.get(v,v)}" for v in variabel_str]))
    with col_r:
        if non_linear_drv:
            st.warning(f"🔴 **Non-Linear Driver** (tidak signifikan linear, FI tinggi):\n" +
                       "\n".join([f"- {LABEL_MAP.get(v,v)}" for v in non_linear_drv]))
        if variabel_minor:
            st.markdown(f"⚪ **Variabel Minor** (FI rendah, tidak signifikan):\n" +
                        "\n".join([f"- {LABEL_MAP.get(v,v)}" for v in variabel_minor]))


# ═══════════════════════════════════════════════════════════
# PAGE: PETA IPM
# ═══════════════════════════════════════════════════════════
elif page == "🗺️ Peta IPM":
    st.markdown('<div class="section-header">🗺️ Peta Choropleth Jawa Timur</div>', unsafe_allow_html=True)

    c_l, c_r = st.columns([1, 2])
    with c_l:
        sel_year_map = st.selectbox("Tahun:", YEARS, index=len(YEARS)-1)
        # Pilih variabel — tampilkan asli (bukan Ln)
        raw_vars = ["IPM", "PDRB_Per_Orang", "Tingkat_Kemiskinan", "Tingkat_Pengangguran",
                    "Angka_Melek_Huruf", "Angka_Partisipasi_Murni", "Jumlah_Sekolah",
                    "Rasio_Murid_dan_Guru", "Sanitasi_Kesehatan", "Kepadatan_Penduduk",
                    "Keluhan_Kesehatan", "Jumlah_Penduduk"]
        raw_vars = [v for v in raw_vars if v in df.columns]
        sel_var_map = st.selectbox("Variabel:", raw_vars)

    df_map = df[df["Tahun"] == sel_year_map][["Kabupaten_Kota", sel_var_map]].copy()

    fig_map = px.choropleth_mapbox(
        df_map,
        geojson=geojson,
        locations="Kabupaten_Kota",
        featureidkey="properties.name",
        color=sel_var_map,
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=7,
        center={"lat": -7.5, "lon": 112.0},
        opacity=0.75,
        hover_name="Kabupaten_Kota",
        title=f"{sel_var_map.replace('_',' ')} — Jawa Timur {sel_year_map}"
    )
    fig_map.update_layout(height=540, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # Ranking
    st.markdown('<div class="section-header">Ranking</div>', unsafe_allow_html=True)
    df_rank = df_map.sort_values(sel_var_map, ascending=False).reset_index(drop=True)
    df_rank.index += 1
    df_rank.columns = ["Kabupaten/Kota", sel_var_map.replace("_", " ")]
    st.dataframe(df_rank, use_container_width=True, height=400)
