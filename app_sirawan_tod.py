# -*- coding: utf-8 -*-
"""SI-RAWAN TOD: Spatial Decision Support System dislokasi usaha mikro koridor MRT Jakarta.

Jalankan:  streamlit run app_sirawan_tod.py   (data dibaca dari ./data)
"""
import json, os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import folium
from streamlit_folium import st_folium

DATA = Path(__file__).parent / "data"
OMZET, MARGIN = 197.8, 0.58          # Rp juta/UMK/th; 1 - rasio biaya 0,42 (BPS, SE2016 Lanjutan)
INK, ROYAL, CRIMSON, EMERALD, AMBER, SLATE = "#0B2545", "#134074", "#8D0801", "#2B8A3E", "#E85D04", "#64748B"
KELAS_C = {"Sangat tinggi": "#8D0801", "Tinggi": "#E85D04", "Sedang": "#F48C06", "Rendah": "#FED9B7"}
KL_C = {1: "#8D0801", 2: "#134074", 3: "#2B8A3E"}
KL_N = {1: "Klaster 1 · Darurat perdagangan rakyat", 2: "Klaster 2 · Komersial transisi", 3: "Klaster 3 · Penyangga parsial"}
JALUR_GAYA = {"Fase 1": ("#0F172A", 5, None), "Fase 2": ("#7B3294", 4, "10,8"), "Timur–Barat": ("#2E75B6", 4, "2,8")}
TIPE_ST = {"Lebak Bulus": "Permukiman selatan", "Fatmawati": "Permukiman selatan", "Cipete Raya": "Permukiman selatan",
           "Haji Nawi": "Permukiman selatan", "Blok A": "Permukiman selatan", "Blok M": "Sub-pusat", "ASEAN": "Sub-pusat",
           "Senayan": "Sub-pusat", "Istora": "Sub-pusat", "Bendungan Hilir": "CBD utama", "Setiabudi": "CBD utama",
           "Dukuh Atas": "CBD utama", "Bundaran HI": "CBD utama"}
IGK_AMBANG, DENS_AMBANG = 13.9, 21.9
rp = lambda v, n=1: f"{v:,.{n}f}".replace(",", "X").replace(".", ",").replace("X", ".")
FONT = dict(family="IBM Plex Sans, Segoe UI, sans-serif", size=12, color="#0F172A")

st.set_page_config(page_title="SI-RAWAN TOD | Bappeda DKI Jakarta", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, [class*="css"], .stApp, button, input, select, textarea { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif !important; }
.stApp { background: #F8FAFC; }
.block-container { padding-top: 1.1rem; padding-bottom: 1rem; max-width: 1500px; }
h1, h2, h3, h4 { color: #0F172A; letter-spacing: -.01em; }
.topbar { display:flex; justify-content:space-between; align-items:flex-end; border-bottom:2px solid #0B2545; padding-bottom:8px; margin-bottom:10px; gap:16px; flex-wrap:wrap; }
.topbar .t { font-size:22px; font-weight:700; color:#0B2545; }
.topbar .s { font-size:12.5px; color:#475569; }
.topbar .src { font-family:'IBM Plex Mono', monospace; font-size:11px; color:#64748B; text-align:right; }
.kpi { background:#FFF; border:1px solid #E2E8F0; border-radius:6px; padding:9px 12px; }
.kpi .l { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:#64748B; }
.kpi .v { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; color:#0F172A; line-height:1.2; }
.kpi .d { font-size:11.5px; color:#475569; }
.kpi.neg .v { color:#8D0801; } .kpi.pos .v { color:#2B8A3E; }
.box { background:#FFF; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px; }
.lbl { font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:#64748B; font-weight:600; margin:6px 0 2px; }
.opd { border-left:3px solid #2B8A3E; background:#F1F5F9; padding:6px 9px; margin:5px 0; font-size:12.5px; border-radius:3px; }
.opd b { color:#0F172A; }
.msg { background:#FFF; border:1px solid #E2E8F0; border-left:4px solid #134074; padding:9px 12px; border-radius:4px; font-size:13.5px; }
.small { font-size:11.5px; color:#64748B; }
div[data-testid="stMetric"] { background:#FFF; border:1px solid #E2E8F0; border-radius:6px; padding:8px 10px; }
div[data-testid="stMetricValue"] { font-size:20px !important; } div[data-testid="stMetricValue"] > div { white-space:normal !important; overflow:visible !important; }
div[data-testid="stMetricLabel"] p { font-size:11.5px !important; }
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }
div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * { white-space:normal !important; overflow:visible !important; text-overflow:clip !important; height:auto !important; }
section[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E2E8F0; }
.stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { font-weight:600; font-size:14px; padding:8px 14px; }
</style>""", unsafe_allow_html=True)


# ================================================================== data
@st.cache_data
def muat(_versi=None, versi=0.0):
    geo = json.load(open(DATA / "kelurahan_dki.geojson", encoding="utf-8"))
    jalur = json.load(open(DATA / "jalur_mrt.geojson", encoding="utf-8"))
    a = pd.read_csv(DATA / "kelurahan_atribut.csv", index_col=0)
    a["kl"] = a.Klaster.str.extract(r"Klaster (\d)").astype(float)
    a["rawan"] = a.koridor.notna()
    a["rank"] = a.ikk.rank(ascending=False, method="first")
    a["pct_pro"] = a.Proyeksi_Hilang / a.w24 * 100
    a = a.join(pd.read_csv(DATA / "hunian_kelurahan.csv", index_col=0))   # Indeks Tekanan Hunian (DPRKP + satelit)
    stn = pd.read_csv(DATA / "stasiun_fase1.csv"); stn["tipe"] = stn.stasiun.map(TIPE_ST)
    pst = pd.read_csv(DATA / "dampak_per_stasiun.csv")
    es = pd.read_csv(DATA / "event_study_sel.csv")
    cin = pd.read_csv(DATA / "profil_cincin.csv")
    val = pd.read_csv(DATA / "kmeans_validasi.csv")
    seri = pd.read_csv(DATA / "seri_warung_koridor.csv")
    return geo, jalur, a, stn, pst, es, cin, val, seri


geo, jalur, A, STN, PST, ES, CIN, VAL, SERI = muat(versi=max(p.stat().st_mtime for p in DATA.iterdir()))
R69 = A[A.rawan].copy()

# ================================================================== kontrol global
with st.sidebar:
    st.markdown("**Kontrol global**")
    kor = st.radio("Koridor", ["Semua koridor", "MRT Fase 1 (evaluasi aktual)", "MRT Fase 2 (konstruksi)", "MRT Timur–Barat (rencana)"], key="kor")
    kota_opsi = sorted(A.kota.dropna().unique())
    kota = st.multiselect("Kota administrasi", kota_opsi, default=[], placeholder="Semua kota", key="kota")
    kec_opsi = sorted(A[A.kota.isin(kota) if kota else A.kota.notna()].kecamatan.dropna().unique())
    kec = st.multiselect("Kecamatan", kec_opsi, default=[], placeholder="Semua kecamatan", key="kec")
    mode = st.radio("Mode angka", ["Unit fisik", "Nilai ekonomi", "Persentase"], key="mode", horizontal=False)
    st.markdown("<div class='small'>Unit fisik: jumlah warung. Nilai ekonomi: omzet Rp miliar per tahun (Rp197,8 juta per usaha). "
                "Persentase: terhadap warung 2024 di kelurahan.</div>", unsafe_allow_html=True)


def saring(df):
    m = pd.Series(True, index=df.index)
    if kor.startswith("MRT Fase 1"): m &= df.fase1
    elif kor.startswith("MRT Fase 2"): m &= df.koridor.eq("Fase 2")
    elif kor.startswith("MRT Timur"): m &= df.koridor.eq("Timur–Barat")
    else: m &= df.fase1 | df.rawan
    if kota: m &= df.kota.isin(kota)
    if kec: m &= df.kecamatan.isin(kec)
    return df[m]


def tampil(warung, basis=None, n=0):
    """Konversi jumlah warung ke mode angka aktif."""
    if mode == "Nilai ekonomi": return f"Rp{rp(warung * OMZET / 1000)} M"
    if mode == "Persentase" and basis: return f"{rp(warung / basis * 100)}%"
    return f"{rp(warung, n)} warung"


U = saring(A)

# ================================================================== parameter kebijakan (dipakai lintas workspace)
ss = st.session_state
for k_, v_ in {"beta_opt": "800 m (β = −1,014**)", "beta_custom": -1.014, "klb": 15, "jak": 35, "kred": 0, "biaya": 12.0,
               "eff_j": 0.75, "eff_k": 0.40, "cap": 0.92, "pagu": 12.1, "simmode": "A · Simulator tuas kebijakan"}.items():
    ss.setdefault(k_, v_)
BETA_OPT = {"500 m (β = −1,707***)": -1.707, "800 m (β = −1,014**)": -1.014, "1.000 m (β = −0,922***)": -0.922, "Kustom": None}


def beta_aktif():
    b = BETA_OPT[ss.beta_opt]; return ss.beta_custom if b is None else b


def tingkat_mitigasi():
    r = 1 - (1 - ss.klb / 100) * (1 - ss.jak / 100 * ss.eff_j) * (1 - ss.kred / 100 * ss.eff_k)
    return min(ss.cap, r)


def simulasi(df):
    d = df[df.rawan].copy(); b = beta_aktif()
    skala = (1 - np.exp(b * d.paparan_baru)) / (1 - np.exp(-1.014 * d.paparan_baru))
    d["bau"] = np.minimum(d.w24, d.Proyeksi_Hilang * skala)
    if ss.simmode.startswith("A"):
        d["sel"] = np.round(np.round(d.bau) * tingkat_mitigasi())  # dibulatkan per kelurahan, sama dengan simulasi di naskah
    else:
        d = d.sort_values("ikk", ascending=False)
        biaya_full = d.bau * ss.cap * ss.biaya / 1000
        d["didanai"] = biaya_full.cumsum() <= ss.pagu
        d["sel"] = np.where(d.didanai, d.bau * ss.cap, 0.0)
    d["pasca"] = d.bau - d.sel
    return d


SIM = simulasi(U)

# ================================================================== header + KPI
st.markdown(f"""<div class="topbar"><div><div class="t">SI-RAWAN TOD</div>
<div class="s">Spatial decision support system dislokasi usaha mikro koridor MRT Jakarta · Bappeda Provinsi DKI Jakarta</div></div>
<div class="src">Open Buildings 2.5D 2016–2023 · Podes 2014–2024 · DPRKP (Satu Data Jakarta) · OSM · GADM 4.1<br>Cakupan aktif: {len(U)} kelurahan · {kor}</div></div>""", unsafe_allow_html=True)
f1u = U[U.fase1]; bau, sel = SIM.bau.sum(), SIM.sel.sum()
k = st.columns(5)
kp = [("Bangunan &lt; 500 m jalur Fase 1", "+4,8–5,7%", "tinggi +4,34%** · tapak −3,85%", ""),
      ("Warung hilang, Fase 1 (estimasi)", tampil(f1u.hilang_f1.sum(), f1u.w24.sum()) if len(f1u) else "–", f"{len(f1u)} kelurahan · β −1,141***", "neg"),
      ("Kelurahan rawan jalur baru", f"{int(U.rawan.sum())}", f"{int((U.koridor == 'Fase 2').sum())} Fase 2 · {int((U.koridor == 'Timur–Barat').sum())} Timur–Barat", ""),
      ("Terancam tanpa kebijakan", tampil(bau, U[U.rawan].w24.sum()) if len(SIM) else "–", f"β = {rp(beta_aktif(), 3)}", "neg"),
      ("Dipertahankan oleh kebijakan", tampil(sel, U[U.rawan].w24.sum()) if len(SIM) else "–", (f"tingkat mitigasi {rp(tingkat_mitigasi() * 100)}%" if ss.simmode.startswith("A") else f"{rp(sel / bau * 100 if bau else 0)}% dari terancam"), "pos")]
for c_, (l, v, d_, cls) in zip(k, kp):
    c_.markdown(f"<div class='kpi {cls}'><div class='l'>{l}</div><div class='v'>{v}</div><div class='d'>{d_}</div></div>", unsafe_allow_html=True)

ws1, ws2, ws3, ws4 = st.tabs(["Peta intelijen spasial", "Simulator & optimasi anggaran", "Diagnostik klaster & ambang", "Laboratorium bukti Fase 1"])


def plot_base(fig, h=360, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="white", plot_bgcolor="white", font=FONT,
                      legend=dict(orientation="h", y=-0.18, font=dict(size=11)), **kw)
    fig.update_xaxes(gridcolor="#EEF2F6", zerolinecolor="#94A3B8"); fig.update_yaxes(gridcolor="#EEF2F6", zerolinecolor="#94A3B8")
    return fig


def skala_warna(v, lo, hi, palet=("#F1F5F9", "#FDBA74", "#E85D04", "#8D0801")):
    if pd.isna(v): return "#E9EDF2"
    t = min(1, max(0, (v - lo) / (hi - lo if hi > lo else 1))) * (len(palet) - 1); i = min(int(t), len(palet) - 2); f = t - i
    a_, b_ = [np.array([int(c[j:j + 2], 16) for j in (1, 3, 5)]) for c in (palet[i], palet[i + 1])]
    return "#%02x%02x%02x" % tuple((a_ + (b_ - a_) * f).astype(int))


# ================================================================== WORKSPACE 1
with ws1:
    LAYERS = ["Indeks kerawanan", "Tipologi klaster", "Kepadatan warung / 1.000 KK", "Intensitas komersial (IGK)", "Dampak pasca-kebijakan",
              "Tekanan hunian", "RW kumuh belum ditangani 2025"]
    lay = st.radio("Layer", LAYERS, horizontal=True, key="layer", label_visibility="collapsed")
    kiri, kanan = st.columns([1.55, 1.0])
    postpct = (SIM.pasca / SIM.w24 * 100) if len(SIM) else pd.Series(dtype=float)

    def nilai(k_):
        r = A.loc[k_]
        if lay == "Indeks kerawanan":
            if r.rawan: return KELAS_C[r.kelas], f"{r.kelas} · indeks {rp(r.ikk, 2)}"
            if r.fase1: return skala_warna(r.hilang_f1 / r.w24 * 100, 0, 80), f"Fase 1 · estimasi hilang {rp(r.hilang_f1, 0)} warung"
            return None, ""
        if lay == "Tipologi klaster":
            return (KL_C[int(r.kl)], KL_N[int(r.kl)]) if r.rawan else (None, "")
        if lay == "Kepadatan warung / 1.000 KK":
            return skala_warna(r.dens24, 0, 40), f"{rp(r.dens24)} warung per 1.000 KK"
        if lay == "Intensitas komersial (IGK)":
            return skala_warna(r.igk24, 0, 50, ("#F1F5F9", "#93C5FD", "#2563EB", "#0B2545")), f"IGK 2024 {rp(r.igk24)}"
        if lay == "Tekanan hunian":
            if pd.isna(r.ith100): return None, ""
            pr_ = f" · {r.prioritas}" if str(r.prioritas).startswith("Prioritas") else ""
            return skala_warna(r.ith100, 0, 80, ("#F1F5F9", "#C4B5FD", "#7C3AED", "#3B0764")), f"Indeks tekanan hunian {rp(r.ith100)}{pr_}"
        if lay == "RW kumuh belum ditangani 2025":
            if pd.isna(r.rw_belum25) or r.rw_belum25 == 0: return None, ""
            return skala_warna(r.rw_belum25, 0, 8, ("#FDE68A", "#F59E0B", "#B45309", "#78350F")), f"{rp(r.rw_belum25, 0)} RW kumuh belum ditangani (dari {rp(r.rw_kumuh17, 0)} RW, Pergub 90/2018)"
        if k_ in postpct.index:
            v = postpct[k_]; return skala_warna(v, 0, 60, ("#2B8A3E", "#FACC15", "#E85D04", "#8D0801")), f"Hilang pasca-kebijakan {rp(v)}% (BAU {rp(SIM.at[k_, 'bau'] / A.at[k_, 'w24'] * 100)}%)"
        return None, ""

    pil_opsi = list(U.sort_values(["rawan", "ikk"], ascending=[False, False]).index)
    if not pil_opsi:
        st.info("Tidak ada kelurahan pada kombinasi filter ini. Ubah filter koridor atau wilayah di panel kiri.")
    else:
        if ss.get("kel1") not in pil_opsi: ss["kel1"] = "SENEN|BUNGUR" if "SENEN|BUNGUR" in pil_opsi else pil_opsi[0]
        with kiri:
            m = folium.Map(location=[-6.205, 106.83], zoom_start=12, tiles="OpenStreetMap", control_scale=True)
            props = {}
            for f in geo["features"]:
                k_ = f["properties"]["kunci"]
                if k_ in A.index and k_ in U.index:
                    c_, t_ = nilai(k_); props[k_] = (c_, t_)
                    f["properties"]["info"] = t_ or "Di luar layer ini"
                else:
                    props[k_] = (None, ""); f["properties"]["info"] = "Di luar filter aktif"

            def gaya(f):
                k_ = f["properties"]["kunci"]; c_ = props.get(k_, (None, ""))[0]; on = k_ in (ss.kel1, ss.get("kel2"))
                if c_ is None: return {"fillColor": "#E2E8F0", "color": "#CBD5E1", "weight": 0.4, "fillOpacity": 0.25}
                return {"fillColor": c_, "color": "#0F172A" if on else "#FFFFFF", "weight": 3.2 if on else 0.7, "fillOpacity": 0.85}
            folium.GeoJson(geo, style_function=gaya, highlight_function=lambda f: {"weight": 2.5, "color": "#0F172A"},
                           tooltip=folium.GeoJsonTooltip(fields=["kelurahan", "kecamatan", "info"], aliases=["", "", ""], labels=False)).add_to(m)
            for f in jalur["features"]:
                c_, w_, d_ = JALUR_GAYA[f["properties"]["jalur"]]
                for line in f["geometry"]["coordinates"]:
                    folium.PolyLine([(y, x) for x, y in line], color=c_, weight=w_, dash_array=d_, tooltip=f"MRT {f['properties']['jalur']}").add_to(m)
            for r in STN.itertuples():
                folium.CircleMarker((r.lat, r.lon), radius=4.5, color="#0F172A", weight=2, fill=True, fill_color="#FFFFFF", fill_opacity=1, tooltip=r.stasiun).add_to(m)
            out = st_folium(m, height=590, use_container_width=True, returned_objects=["last_active_drawing"], key="peta1")
            klik = (out or {}).get("last_active_drawing")
            if klik:
                kk_ = klik.get("properties", {}).get("kunci")
                if kk_ in pil_opsi and kk_ != ss.kel1 and kk_ != ss.get("_klik_terakhir"):
                    ss["kel1"] = kk_; ss["_klik_terakhir"] = kk_; st.rerun()
            legenda = {"Indeks kerawanan": "Kelas kuartil indeks kerawanan; Fase 1 diwarnai menurut estimasi persentase warung hilang.",
                       "Tipologi klaster": "Merah = klaster 1, biru = klaster 2, hijau = klaster 3 (K-Means, k = 3).",
                       "Kepadatan warung / 1.000 KK": "Skala 0–40 warung per 1.000 keluarga (2024).",
                       "Intensitas komersial (IGK)": "Skala IGK 0–50 (2024); makin gelap makin komersial.",
                       "Dampak pasca-kebijakan": "Persentase warung yang tetap hilang setelah kebijakan aktif di Workspace 2: hijau rendah, merah tinggi.",
                       "Tekanan hunian": "Indeks 0–100: keparahan RW kumuh (Pergub 90/2018), RW kumuh belum ditangani 2025 (DPRKP), dan kesesakan (luas lantai satelit per keluarga).",
                       "RW kumuh belum ditangani 2025": "Jumlah RW kumuh yang belum ditangani per 2025 (Satu Data Jakarta, DPRKP); abu-abu = tidak ada RW kumuh tersisa."}[lay]
            st.markdown(f"<div class='small'>{legenda} Klik poligon untuk memilih kelurahan.</div>", unsafe_allow_html=True)

        with kanan:
            fmt_k = lambda k_: f"{A.at[k_, 'kelurahan']} · {A.at[k_, 'koridor_all']}" + (f" · #{int(A.at[k_, 'rank'])}" if A.at[k_, "rawan"] else "")
            st.selectbox("Kelurahan", pil_opsi, key="kel1", format_func=fmt_k)
            r = A.loc[ss.kel1]
            st.markdown(f"<div class='lbl'>{r.kecamatan} · {r.kota}</div>", unsafe_allow_html=True)
            if isinstance(r.flag_data, str) and r.flag_data:
                st.markdown(f"<div class='msg' style='border-left-color:#E85D04'><b>Catatan kualitas data:</b> {r.flag_data} "
                            f"(warung Podes 2014/2018/2021/2024: {rp(r.w14, 0)} / {rp(r.w18, 0)} / {rp(r.w21, 0)} / {rp(r.w24, 0)}). "
                            "Proyeksi kelurahan ini perlu diverifikasi di lapangan.</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Warung 2024", rp(r.w24, 0), f"Podes 2018: {rp(r.w18, 0)}" if r.w18 == r.w18 else None, delta_color="off")
            c2.metric("Warung / 1.000 KK", rp(r.dens24), f"IGK {rp(r.igk24)}", delta_color="off")
            if r.rawan:
                c3.metric("Proyeksi hilang", tampil(r.Proyeksi_Hilang, r.w24), f"paparan {rp(r.paparan_baru * 100, 0)}%", delta_color="off")
            elif r.fase1:
                c3.metric("Estimasi hilang (aktual)", tampil(r.hilang_f1, r.w24), f"paparan Fase 1 {rp(r.paparan1 * 100, 0)}%", delta_color="off")
            if not pd.isna(r.ith100):
                h1, h2, h3 = st.columns(3)
                h1.metric("RW kumuh 2017", rp(r.rw_kumuh17, 0), f"{rp(r.klasifikasi_berat, 0)} kelas berat", delta_color="off")
                h2.metric("Belum ditata", rp(r.rw_belum25, 0), r.prioritas if str(r.prioritas).startswith("Prioritas") else None, delta_color="off")
                h3.metric("Lantai/KK", f"{rp(r.lantai_kk, 0)} m²", f"indeks hunian {rp(r.ith100)}", delta_color="off")
            if r.rawan:
                st.markdown("<div class='lbl'>Anatomi skor kerawanan (z-score terhadap 69 kelurahan rawan)</div>", unsafe_allow_html=True)
                komp = pd.DataFrame({"Komponen": ["Paparan radius 800 m", "Skala proyeksi warung hilang (log)", "Kepadatan warung / 1.000 KK"],
                                     "z": [r.z_paparan, r.z_proyeksi, r.z_kepadatan]})
                fig = go.Figure(go.Bar(y=komp.Komponen, x=komp.z, orientation="h", marker_color=[CRIMSON if v > 0 else ROYAL for v in komp.z],
                                       text=[rp(v, 2) for v in komp.z], textposition="outside"))
                fig.add_vline(x=0, line_color="#0F172A", line_width=1)
                fig.add_annotation(x=0, y=1.12, yref="paper", text="rata-rata 69 kelurahan", showarrow=False, font=dict(size=10, color=SLATE))
                plot_base(fig, 190, showlegend=False, title=dict(text=f"Indeks {rp(r.ikk, 2)} = jumlah ketiga z-score · kelas {r.kelas}", font=dict(size=12)))
                fig.update_layout(margin=dict(l=10, r=40, t=40, b=10)); fig.update_xaxes(range=[min(-2.5, komp.z.min() - .6), max(2.5, komp.z.max() + .6)])
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            banding = st.toggle("Bandingkan dengan kelurahan lain", key="banding")
            if banding:
                opsi2 = [x for x in pil_opsi if x != ss.kel1]
                if ss.get("kel2") not in opsi2: ss["kel2"] = "TAMANSARI|GLODOK" if "TAMANSARI|GLODOK" in opsi2 else opsi2[0]
                st.selectbox("Kelurahan pembanding", opsi2, key="kel2", format_func=fmt_k)
                r2 = A.loc[ss.kel2]
                dims = [("Paparan jalur", "paparan_baru"), ("Warung 2024", "w24"), ("Kepadatan warung", "dens24"), ("IGK 2024", "igk24"),
                        ("Penurunan warung 2014–24", None), ("Proyeksi hilang (%)", "pct_pro")]
                def persentil(row, col):
                    if col is None:
                        s_ = (1 - A.w24 / A.w14).rank(pct=True); return float(s_.get(row.name, np.nan)) * 100
                    s_ = A[col].rank(pct=True); return float(s_.get(row.name, np.nan)) * 100 if not pd.isna(row.get(col)) else 0
                fig = go.Figure()
                for row, c_ in ((r, CRIMSON), (r2, ROYAL)):
                    v = [persentil(row, col) for _, col in dims]
                    fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=[d[0] for d in dims] + [dims[0][0]], fill="toself", name=row.kelurahan,
                                                  line=dict(color=c_), opacity=0.7))
                plot_base(fig, 300, polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="", tickfont=dict(size=9)), bgcolor="white"),
                          title=dict(text="Persentil terhadap 260 kelurahan DKI", font=dict(size=12)))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                tb = pd.DataFrame({"Indikator": ["Warung 2024", "Warung / 1.000 KK", "IGK 2024", "Paparan jalur (%)", "Proyeksi hilang", "Klaster"],
                                   r.kelurahan: [rp(r.w24, 0), rp(r.dens24), rp(r.igk24), rp((r.paparan_baru if r.rawan else r.paparan1) * 100, 0),
                                                 rp(r.Proyeksi_Hilang, 0) if r.rawan else "–", KL_N.get(int(r.kl), "–") if r.rawan else "–"],
                                   r2.kelurahan: [rp(r2.w24, 0), rp(r2.dens24), rp(r2.igk24), rp((r2.paparan_baru if r2.rawan else r2.paparan1) * 100, 0),
                                                  rp(r2.Proyeksi_Hilang, 0) if r2.rawan else "–", KL_N.get(int(r2.kl), "–") if r2.rawan else "–"]})
                st.dataframe(tb, hide_index=True, use_container_width=True)
            OPD = {1: ["PPKUKM: sensus by name by address dan lokasi usaha sementara sebelum penetapan lokasi konstruksi",
                       "CKTRP: perlindungan permukiman padat pada cincin 500–1.000 m",
                       "Bank DKI: kredit modal kerja bunga rendah untuk warung terdampak"],
                   2: ["CKTRP & PT MRT: kuota 10–15% ruang ritel mikro lantai dasar dalam syarat insentif KLB",
                       "PPKUKM: kurasi pedagang lokal ke ruang ritel baru", "Bapenda: pungutan pelampauan KLB untuk subsidi sewa"],
                   3: ["Dishub/Jaklingko: rute pengumpan menembus permukiman", "PPKUKM: digitalisasi warung (Jakpreneur)",
                       "Bappeda: pemantauan tahunan indeks kerawanan"]}
            pilihan = [r] + ([A.loc[ss.kel2]] if banding else [])
            cols = st.columns(len(pilihan))
            for c_, row in zip(cols, pilihan):
                with c_:
                    st.markdown(f"<div class='lbl'>Strategi OPD · {row.kelurahan}</div>", unsafe_allow_html=True)
                    if row.rawan:
                        st.markdown("".join(f"<div class='opd'>{t}</div>" for t in OPD[int(row.kl)]), unsafe_allow_html=True)
                        if str(row.prioritas).startswith("Prioritas"):
                            st.markdown(f"<div class='opd' style='border-left-color:#7C3AED'>DPRKP: {row.prioritas.lower()} hunian TOD — tangani "
                                        f"{rp(row.rw_belum25, 0)} RW kumuh sebelum konstruksi, siapkan rusun/penataan in-situ ≤ 800 m stasiun, "
                                        "kuota FPPR lewat SIRUKIM</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='opd'>Kelurahan koridor Fase 1: evaluasi pasca-operasi, pemulihan usaha mikro, dan pemantauan.</div>", unsafe_allow_html=True)

# ================================================================== WORKSPACE 2
with ws2:
    st.radio("Mode simulasi", ["A · Simulator tuas kebijakan", "B · Optimasi pagu anggaran LVC"], horizontal=True, key="simmode")
    kiri, kanan = st.columns([0.95, 2.05])
    with kiri:
        st.markdown("<div class='lbl'>Elastisitas radius dampak</div>", unsafe_allow_html=True)
        st.radio("Radius", list(BETA_OPT), key="beta_opt", label_visibility="collapsed")
        if ss.beta_opt == "Kustom":
            st.slider("β kustom", -2.5, -0.2, step=0.05, key="beta_custom")
        if ss.simmode.startswith("A"):
            st.slider("Kuota ruang ritel mikro via insentif KLB, zona 0–500 m (%)", 0, 50, key="klb")
            st.slider("Proteksi pra-konstruksi Jakpreneur & relokasi sementara (%)", 0, 100, key="jak")
            st.slider("Kredit modal kerja Bank DKI & rantai pasok BUMD pangan (%)", 0, 100, key="kred")
        else:
            st.slider("Pagu dana LVC tersedia (Rp miliar/tahun)", 1.0, 50.0, step=0.1, key="pagu")
        st.number_input("Biaya intervensi per warung dipertahankan (Rp juta/tahun)", 1.0, 100.0, step=0.5, key="biaya")
        with st.expander("Asumsi model"):
            st.slider("Efektivitas proteksi Jakpreneur", 0.0, 1.0, step=0.05, key="eff_j")
            st.slider("Efektivitas kredit & rantai pasok", 0.0, 1.0, step=0.05, key="eff_k")
            st.slider("Batas atas tingkat mitigasi", 0.5, 1.0, step=0.01, key="cap")
            st.markdown("<div class='small'>Tingkat mitigasi = min{batas; 1 − (1 − KLB)(1 − Jakpreneur × efektivitas)(1 − kredit × efektivitas)}. "
                        "Mode B: kelurahan didanai penuh (mitigasi = batas atas) berurutan dari indeks kerawanan tertinggi sampai pagu habis. "
                        "Seluruh efektivitas dan biaya adalah asumsi skenario.</div>", unsafe_allow_html=True)
    with kanan:
        if not len(SIM):
            st.info("Filter aktif tidak memuat kelurahan jalur baru (Fase 2 atau Timur–Barat).")
        else:
            om, dana = SIM.sel.sum() * OMZET / 1000, SIM.sel.sum() * ss.biaya / 1000
            if ss.simmode.startswith("B"):
                n = int(SIM.didanai.sum())
                st.markdown(f"<div class='msg'>Dengan pagu <b>Rp{rp(ss.pagu)} miliar</b> per tahun, Bappeda dapat mendanai penuh <b>{n} kelurahan prioritas</b>, "
                            f"mempertahankan <b>{rp(SIM.sel.sum(), 0)} warung</b> dan omzet <b>Rp{rp(om)} miliar</b> per tahun "
                            f"(dana terpakai Rp{rp(dana)} miliar; BCR berbasis laba <b>{rp(om * MARGIN / dana if dana else 0)}×</b>). "
                            f"Karena biaya dan manfaat per warung sama di semua kelurahan, prioritas mengikuti indeks kerawanan.</div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Terancam tanpa kebijakan", tampil(SIM.bau.sum(), SIM.w24.sum()))
            m2.metric("Dipertahankan", tampil(SIM.sel.sum(), SIM.w24.sum()), f"{rp(tingkat_mitigasi() * 100 if ss.simmode.startswith(chr(65)) else SIM.sel.sum() / SIM.bau.sum() * 100)}%")
            m3.metric("Dana dibutuhkan", f"Rp{rp(dana)} M/th")
            m4.metric("BCR berbasis laba", f"{rp(om * MARGIN / dana if dana else 0)}×", f"omzet {rp(om / dana if dana else 0)}×", delta_color="off")
            # lintasan aktual 2014-2024 + skenario 2027/2030
            fig = go.Figure()
            warna_seri = {"Koridor Fase 1": INK, "Koridor Fase 2": "#7B3294", "Koridor Timur–Barat": "#2E75B6", "Kelurahan lainnya": "#94A3B8"}
            for g_, d_ in SERI.groupby("kelompok"):
                fig.add_trace(go.Scatter(x=d_.tahun, y=d_.indeks, name=f"{g_} (Podes)", mode="lines+markers", line=dict(color=warna_seri[g_], width=2.2)))
            base24 = SIM.w24.sum(); i24 = 100.0
            lv = lambda hilang: i24 * (1 - hilang / base24)
            th = [2024, 2027, 2030]
            fig.add_trace(go.Scatter(x=th, y=[i24, lv(SIM.bau.sum() * .5), lv(SIM.bau.sum())], name="Jalur baru tanpa kebijakan (skenario)",
                                     mode="lines+markers", line=dict(color=CRIMSON, dash="dash", width=2.4), yaxis="y2"))
            fig.add_trace(go.Scatter(x=th, y=[i24, lv(SIM.pasca.sum() * .5), lv(SIM.pasca.sum())], name="Jalur baru dengan kebijakan (skenario)",
                                     mode="lines+markers", line=dict(color=EMERALD, width=2.8), yaxis="y2"))
            fig.add_vrect(x0=2024, x1=2030.4, fillcolor="#F1F5F9", line_width=0, layer="below")
            fig.add_annotation(x=2027.2, y=1.04, yref="paper", text="skenario: pola Fase 1 berulang, dampak penuh 2030", showarrow=False, font=dict(size=10, color=SLATE))
            plot_base(fig, 360, title=dict(text="Indeks warung kelontong: aktual 2014–2024 (2014 = 100) dan skenario jalur baru (2024 = 100, sumbu kanan)", font=dict(size=12)),
                      yaxis=dict(title="2014 = 100"), yaxis2=dict(title="2024 = 100", overlaying="y", side="right", range=[40, 105], showgrid=False))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            c1, c2 = st.columns(2)
            with c1:
                kl = SIM.groupby("kl").agg(bau=("bau", "sum"), sel=("sel", "sum"))
                lab = ["Warung terancam"] + [f"Klaster {int(i)}" for i in kl.index] + ["Dipertahankan", "Tetap hilang"]
                src, tgt, val = [], [], []
                for j, (i, r_) in enumerate(kl.iterrows(), start=1):
                    src += [0, j, j]; tgt += [j, len(lab) - 2, len(lab) - 1]; val += [r_.bau, r_.sel, r_.bau - r_.sel]
                fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=lab, pad=14, thickness=14, x=[0.001] + [0.42] * len(kl) + [0.999, 0.999], y=[0.5] + [0.15 + 0.7 * j / max(len(kl) - 1, 1) for j in range(len(kl))] + [0.25, 0.75], color=[SLATE] + [KL_C[int(i)] for i in kl.index] + [EMERALD, CRIMSON]),
                                          link=dict(source=src, target=tgt, value=val, color="rgba(100,116,139,0.25)"),
                                          valueformat=",.0f", valuesuffix=" warung"))
                plot_base(fig, 330, title=dict(text="Aliran warung terancam per klaster", font=dict(size=12)))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                tarif = st.number_input("Kontribusi LVC dari tambahan 2,05 juta m² lantai (Rp ribu/m²/tahun)", 0.0, 500.0, 0.0, 5.0, key="tarif")
                if tarif:
                    pot = 2.05e6 * tarif * 1000 / 1e9
                    st.markdown(f"<div class='small'>Potensi LVC Rp{rp(pot)} miliar/th vs kebutuhan Rp{rp(dana)} miliar/th: "
                                f"{'cukup' if pot >= dana else 'belum cukup'} ({rp(pot / dana * 100 if dana else 0)}%). Tarif per m² adalah input pengguna.</div>", unsafe_allow_html=True)
            with c2:
                t15 = SIM.sort_values("bau", ascending=False).head(15).iloc[::-1]
                fig = go.Figure()
                for _, r_ in t15.iterrows():
                    fig.add_trace(go.Scatter(x=[r_.pasca, r_.bau], y=[r_.kelurahan] * 2, mode="lines", line=dict(color="#CBD5E1", width=4), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=t15.bau, y=t15.kelurahan, mode="markers", name="Tanpa kebijakan", marker=dict(color=CRIMSON, size=10)))
                fig.add_trace(go.Scatter(x=t15.pasca, y=t15.kelurahan, mode="markers", name="Dengan kebijakan", marker=dict(color=ROYAL, size=10)))
                plot_base(fig, 420, title=dict(text="15 kelurahan terbesar: warung hilang sebelum vs sesudah", font=dict(size=12)), xaxis=dict(title="warung"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ================================================================== WORKSPACE 3
with ws3:
    R = saring(A)[lambda d: d.rawan].copy()
    if not len(R):
        st.info("Filter aktif tidak memuat kelurahan jalur baru.")
    else:
        c1, c2 = st.columns([1.45, 1.0])
        with c1:
            warnai = st.radio("Warnai titik menurut", ["Klaster K-Means", "Koridor", "Kota administrasi"], horizontal=True, key="warnai")
            R["q"] = np.select([(R.igk14 > IGK_AMBANG) & (R.dens14 > DENS_AMBANG), (R.igk14 <= IGK_AMBANG) & (R.dens14 > DENS_AMBANG),
                                (R.igk14 > IGK_AMBANG) & (R.dens14 <= DENS_AMBANG)], ["I", "II", "III"], "IV")
            xm, ym = max(R.igk14.max() * 1.08, 30), max(R.dens14.max() * 1.08, 40)
            fig = go.Figure()
            for x0, x1, y0, y1, col, lab in [(IGK_AMBANG, xm, DENS_AMBANG, ym, "rgba(141,8,1,.07)", "I · Dislokasi kritis"),
                                             (0, IGK_AMBANG, DENS_AMBANG, ym, "rgba(232,93,4,.07)", "II · Kampung padat pra-komersial"),
                                             (IGK_AMBANG, xm, 0, DENS_AMBANG, "rgba(19,64,116,.07)", "III · Sentra komersial / CBD tua"),
                                             (0, IGK_AMBANG, 0, DENS_AMBANG, "rgba(100,116,139,.05)", "IV · Risiko rendah / penyangga")]:
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=col, line_width=0, layer="below")
                fig.add_annotation(x=x1 - (x1 - x0) * .03, y=y1 - (y1 - y0) * .03, text=f"<b>{lab.split(chr(32))[0]}</b>", showarrow=False, xanchor="right", yanchor="top", font=dict(size=15, color="#94A3B8"))
            fig.add_vline(x=IGK_AMBANG, line_dash="dash", line_color="#0F172A"); fig.add_hline(y=DENS_AMBANG, line_dash="dash", line_color="#0F172A")
            grp = {"Klaster K-Means": R.kl.map(lambda v: KL_N[int(v)]), "Koridor": R.koridor, "Kota administrasi": R.kota}[warnai]
            pal = [CRIMSON, ROYAL, EMERALD, AMBER, "#7B3294", "#0891B2"]
            cmap = {KL_N[1]: CRIMSON, KL_N[2]: ROYAL, KL_N[3]: EMERALD} if warnai == "Klaster K-Means" else {g_: pal[i % 6] for i, g_ in enumerate(sorted(grp.unique()))}
            for g_ in sorted(grp.unique()):
                d_ = R[grp == g_]
                fig.add_trace(go.Scatter(x=d_.igk14, y=d_.dens14, mode="markers", name=g_, customdata=np.c_[d_.kelurahan, d_.Proyeksi_Hilang, d_.q],
                                         marker=dict(size=7 + np.sqrt(d_.Proyeksi_Hilang) * 1.2, color=cmap[g_], opacity=.78, line=dict(color="white", width=1)),
                                         hovertemplate="<b>%{customdata[0]}</b><br>IGK 2014 %{x:.1f} · %{y:.1f} warung/1.000 KK<br>Kuadran %{customdata[2]} · proyeksi hilang %{customdata[1]:.0f}<extra></extra>"))
            top = R[R.dens14.notna()].sort_values("Proyeksi_Hilang", ascending=False).head(7)
            for _, r_ in top.iterrows():
                fig.add_annotation(x=r_.igk14, y=r_.dens14, text=r_.kelurahan, showarrow=False, yshift=12, font=dict(size=10, color="#0F172A"))
            plot_base(fig, 520, title=dict(text=f"Kuadran ambang efek marginal DiD (IGK 2014 = {rp(IGK_AMBANG)}; kepadatan 2014 = {rp(DENS_AMBANG)})", font=dict(size=12)),
                      xaxis=dict(title="IGK 2014 (0–100)", range=[0, xm]), yaxis=dict(title="Warung per 1.000 KK, 2014", range=[0, ym]))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("<div class='small'>Ambang diukur pada kondisi 2014, sesuai estimasi efek marginal di naskah (Gambar 4.3.1). Kuadran I = dislokasi kritis, II = kampung padat pra-komersial, III = sentra komersial/CBD tua, IV = risiko rendah/penyangga. Kelurahan dengan kode isian Podes 997–999 pada 2014 tidak ditampilkan. "
                        "Dampak paparan terhadap warung signifikan di atas ambang; koefisien interaksinya sendiri tidak signifikan, sehingga ambang dibaca sebagai batas mulai terdeteksinya dampak.</div>", unsafe_allow_html=True)
        with c2:
            q = R.groupby("q").agg(n=("q", "size"), proyeksi=("Proyeksi_Hilang", "sum")).reindex(["I", "II", "III", "IV"]).fillna(0)
            q["anggota"] = [", ".join(R[R.q == i].sort_values("Proyeksi_Hilang", ascending=False).kelurahan.head(6)) for i in q.index]
            q["proyeksi"] = [tampil(v, R[R.q == i].w24.sum()) for i, v in zip(q.index, q.proyeksi)]
            st.markdown("<div class='lbl'>Komposisi kuadran</div>", unsafe_allow_html=True)
            st.dataframe(q.reset_index().rename(columns={"q": "Kuadran", "n": "Kelurahan", "proyeksi": "Proyeksi hilang", "anggota": "Contoh anggota"}),
                         hide_index=True, use_container_width=True)
            st.markdown("<div class='lbl'>Validasi jumlah klaster</div>", unsafe_allow_html=True)
            v_ = VAL.rename(columns={"Inertia (WCSS)": "Inertia"}).copy(); v_["Silhouette"] = v_.Silhouette.round(3); v_["Inertia"] = v_.Inertia.round(1)
            st.dataframe(v_, hide_index=True, use_container_width=True)
            st.markdown("<div class='small'>Silhouette tertinggi pada k = 2; k = 3 dipilih karena memisahkan zona darurat, transisi, dan penyangga.</div>", unsafe_allow_html=True)
        st.markdown("<div class='lbl'>Koordinat paralel: seret pada sumbu untuk menyaring rentang</div>", unsafe_allow_html=True)
        R["omzet"] = R.Proyeksi_Hilang * OMZET / 1000
        fig = go.Figure(go.Parcoords(line=dict(color=R.kl, colorscale=[[0, CRIMSON], [.5, ROYAL], [1, EMERALD]], cmin=1, cmax=3),
                                     dimensions=[dict(label="Klaster", values=R.kl, tickvals=[1, 2, 3]),
                                                 dict(label="Paparan (%)", values=R.paparan_baru * 100), dict(label="Warung / 1.000 KK", values=R.dens24),
                                                 dict(label="IGK 2024", values=R.igk24), dict(label="Proyeksi hilang (%)", values=R.pct_pro),
                                                 dict(label="Omzet (Rp M/th)", values=R.omzet)]))
        plot_base(fig, 330); fig.update_layout(margin=dict(l=50, r=90, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='lbl'>Penjelajah klaster dan rencana aksi</div>", unsafe_allow_html=True)
        AKSI = {1: {"Bappeda": ("Tetapkan zona prioritas dan alokasi dana LVC", "Koordinasi jadwal konstruksi dengan OPD", "Evaluasi indeks tahunan"),
                    "Dinas CKTRP": ("Moratorium konversi permukiman padat 500–1.000 m", "Pengawasan perubahan guna lahan", "Konsolidasi lahan vertikal bersama warga lama"),
                    "Dinas PPKUKM": ("Sensus by name by address seluruh warung", "Lokasi usaha sementara dan pasokan grosir BUMD", "Pendampingan pemulihan usaha"),
                    "Bank DKI": ("Pra-kualifikasi kredit modal kerja", "Penundaan cicilan selama konstruksi", "Kredit ekspansi ke ruang ritel baru"),
                    "PT MRT Jakarta": ("Rencana akses pejalan kaki selama konstruksi", "Jalur akses sementara ke warung", "Ruang ritel stasiun untuk pedagang lokal")},
                2: {"Bappeda": ("Masukkan kuota ritel mikro ke panduan rancang kota", "Pantau kepatuhan pengembang", "Evaluasi pangsa ritel mikro"),
                    "Dinas CKTRP": ("Syarat kuota 10–15% dalam insentif KLB", "Verifikasi desain lantai dasar", "Audit pemanfaatan ruang"),
                    "Dinas PPKUKM": ("Kurasi pedagang lokal", "Pelatihan tata kelola gerai", "Penempatan pedagang ke ruang baru"),
                    "Bank DKI": ("Skema pembiayaan sewa", "Kredit renovasi gerai", "Pembayaran digital"),
                    "PT MRT Jakarta": ("Identifikasi ruang ritel kawasan TOD", "Integrasi akses gerai ke stasiun", "Promosi sentra dagang")},
                3: {"Bappeda": ("Masukkan kelurahan ke pemantauan tahunan", "Sinkronisasi rute pengumpan", "Evaluasi berkala"),
                    "Dinas CKTRP": ("Penataan gang dan ruang usaha", "Perbaikan akses ke stasiun", "Pemeliharaan ruang publik"),
                    "Dinas PPKUKM": ("Digitalisasi warung (Jakpreneur)", "Pelatihan pemasaran daring", "Kemitraan dengan ritel formal"),
                    "Bank DKI": ("Tabungan dan pembayaran digital", "Kredit mikro", "Pembiayaan stok"),
                    "PT MRT Jakarta": ("Informasi rute dan jadwal", "Integrasi tarif dengan pengumpan", "Survei pengguna dari permukiman")}}
        tabs_kl = st.tabs([KL_N[i] for i in (1, 2, 3)])
        for i, tb_ in zip((1, 2, 3), tabs_kl):
            with tb_:
                d_ = R[R.kl == i]
                if not len(d_):
                    st.info("Tidak ada anggota pada filter aktif."); continue
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Kelurahan", len(d_)); a2.metric("Proyeksi hilang", tampil(d_.Proyeksi_Hilang.sum(), d_.w24.sum()))
                a3.metric("Paparan rata-rata", f"{rp(d_.paparan_baru.mean() * 100, 0)}%"); a4.metric("Warung / 1.000 KK", rp(d_.dens24.mean()))
                st.markdown("<div class='small'><b>Anggota:</b> " + ", ".join(d_.sort_values("ikk", ascending=False).kelurahan) + "</div>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([(o, *v) for o, v in AKSI[i].items()], columns=["OPD", "Pra-konstruksi", "Masa konstruksi", "Pasca-operasi"]),
                             hide_index=True, use_container_width=True)

# ================================================================== WORKSPACE 4
with ws4:
    c1, c2 = st.columns([1.5, 1.0])
    with c1:
        f1, f2, f3 = st.columns([1.3, 1.2, .8])
        tipe = f1.multiselect("Tipologi stasiun", ["Permukiman selatan", "Sub-pusat", "CBD utama"], default=["Permukiman selatan", "Sub-pusat", "CBD utama"], key="tipe")
        met = f2.selectbox("Indikator", ["Δ volume bangunan", "Δ tinggi rata-rata", "Δ luas tapak"], key="met")
        ci = f3.checkbox("IK 95%", value=True, key="ci")
        o = {"Δ volume bangunan": "d_vol", "Δ tinggi rata-rata": "d_tinggi", "Δ luas tapak": "d_tapak"}[met]
        stn = STN[STN.tipe.isin(tipe)]
        d_ = PST[PST.outcome == o].set_index("stasiun").reindex(stn.stasiun).dropna()
        warna = [CRIMSON if b - 1.96 * s > 0 else ROYAL if b + 1.96 * s < 0 else "#94A3B8" for b, s in zip(d_.b, d_.se)]
        fig = go.Figure(go.Bar(x=d_.index, y=d_.b * 100, marker_color=warna, error_y=dict(type="data", array=1.96 * d_.se * 100, visible=ci, color="#334155"),
                               customdata=np.c_[d_.se * 100, d_.n_sel, stn.set_index("stasiun").loc[d_.index].tipe],
                               hovertemplate="<b>%{x}</b> (%{customdata[2]})<br>%{y:.1f}% · SE %{customdata[0]:.1f} · %{customdata[1]} sel<extra></extra>"))
        plot_base(fig, 360, showlegend=False, title=dict(text=f"{met} radius 500 m tiap stasiun vs pembanding 2–5 km (%), selatan ke utara", font=dict(size=12)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        semua_st = list(STN.stasiun)
        s1 = st.selectbox("Stasiun A", semua_st, index=semua_st.index("Fatmawati"), key="s1")
        s2 = st.selectbox("Stasiun B", semua_st, index=semua_st.index("Bundaran HI"), key="s2")
        lab3 = {"d_vol": "Volume", "d_tinggi": "Tinggi", "d_tapak": "Luas tapak"}
        fig = go.Figure()
        for s_, c_ in ((s1, CRIMSON), (s2, ROYAL)):
            q_ = PST[PST.stasiun == s_].set_index("outcome").reindex(list(lab3))
            fig.add_trace(go.Bar(x=[lab3[i] for i in q_.index], y=q_.b * 100, name=s_, marker_color=c_,
                                 error_y=dict(type="data", array=1.96 * q_.se * 100, visible=ci, color="#334155")))
        plot_base(fig, 290, barmode="group", title=dict(text="Dekomposisi perubahan bangunan (%)", font=dict(size=12)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        def tafsir(s_):
            q_ = PST[PST.stasiun == s_].set_index("outcome")
            sig = lambda o_: abs(q_.at[o_, "b"] / q_.at[o_, "se"]) > 1.96
            v, t, p = (q_.at[o_, "b"] * 100 for o_ in ("d_vol", "d_tinggi", "d_tapak"))
            if sig("d_vol") and v > 0: arah = "tumbuh signifikan"
            elif sig("d_vol") and v < 0: arah = "tumbuh lebih lambat secara signifikan"
            else: arah = "tidak berbeda signifikan"
            pola = "vertikalisasi (tinggi naik, tapak menyusut)" if t > 0 and p < 0 else "penambahan tapak" if p > 0 else "kejenuhan lahan"
            return f"{s_} ({TIPE_ST[s_]}): volume {arah} ({rp(v)}%), dengan pola {pola}."
        st.markdown(f"<div class='msg'>{tafsir(s1)}<br>{tafsir(s2)}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.15, 1.0])
    with c1:
        ind = ["Volume bangunan (%)", "Tinggi bangunan (%)", "Luas tapak (%)", "IGK (SD)", "Jumlah keluarga (ln)", "Warung kelontong (ln)", "Kantor bank (unit)"]
        pita = ["0–500 m", "500–1.000 m", "1–2 km"]
        fig = make_subplots(rows=2, cols=4, subplot_titles=ind, vertical_spacing=.22, horizontal_spacing=.07)
        for i, nm in enumerate(ind):
            d_ = CIN[CIN.indikator == nm].set_index("pita").reindex(pita)
            col = [("#16A34A" if v > 0 else CRIMSON) if isinstance(s_, str) and s_ else "#CBD5E1" for v, s_ in zip(d_.koefisien.fillna(0), d_.sig.fillna(""))]
            fig.add_trace(go.Bar(x=["inti", "penyangga", "luar"], y=d_.koefisien, marker_color=col, showlegend=False,
                                 text=[f"{rp(v, 2)}{s_ if isinstance(s_, str) else ''}" if not pd.isna(v) else "–" for v, s_ in zip(d_.koefisien, d_.sig)],
                                 textposition="outside", textfont=dict(size=10)), row=i // 4 + 1, col=i % 4 + 1)
        fig.update_annotations(font=dict(size=11))
        plot_base(fig, 430, title=dict(text="Gelombang melintang koridor: inti 0–500 m, penyangga 500–1.000 m, luar 1–2 km", font=dict(size=12)))
        fig.update_yaxes(zeroline=True, zerolinecolor="#475569")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='small'>Hijau = naik signifikan, merah = turun signifikan, abu = tidak signifikan. Uji radius kumulatif warung: −1,707*** (500 m), −1,089*** (800 m), −0,922*** (1.000 m). "
                    "Zona penyangga memperlihatkan bayangan aglomerasi: IGK −2,378** dan kantor bank −18,708**.</div>", unsafe_allow_html=True)
    with c2:
        tampilkan = st.multiselect("Seri event study", list(ES.grup.unique()), default=list(ES.grup.unique()), key="es_seri")
        fig = go.Figure()
        warna_es = {"Fase 1 (<1 km)": INK, "Fase 2 (<1 km)": CRIMSON, "Plasebo Timur-Barat (<1 km)": "#94A3B8"}
        for g_ in tampilkan:
            d_ = pd.concat([ES[ES.grup == g_], pd.DataFrame({"tahun": [2017], "b": [0.0], "se": [0.0]})]).sort_values("tahun")
            fig.add_trace(go.Scatter(x=d_.tahun, y=d_.b * 100, name=g_.replace("Timur-Barat", "Timur–Barat"), mode="lines+markers",
                                     line=dict(color=warna_es[g_], width=2.2), error_y=dict(type="data", array=1.96 * d_.se * 100, visible=ci)))
        fig.add_vline(x=2019.2, line_dash="dash", line_color="#475569"); fig.add_hline(y=0, line_color="#475569")
        fig.add_annotation(x=2019.2, y=1.05, yref="paper", text="Fase 1 beroperasi, Maret 2019", showarrow=False, font=dict(size=10, color=SLATE))
        plot_base(fig, 430, title=dict(text="Event study volume bangunan (% vs pembanding 2–5 km, acuan 2017)", font=dict(size=12)), yaxis=dict(title="%"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='small'>Koefisien 2018 untuk Fase 1 tidak signifikan (tren sejajar sebelum operasi); plasebo Timur–Barat datar setelah 2017. Citra 2016 tidak stabil di sebagian wilayah.</div>", unsafe_allow_html=True)
st.markdown("<div class='small' style='margin-top:8px'>Sumber: Google Open Buildings 2.5D Temporal; Podes 2014–2024 (BPS); BPS Provinsi DKI Jakarta (SE2016 Lanjutan); RW kumuh Pergub 90/2018 dan status 2025 (DPRKP, Satu Data Jakarta); OpenStreetMap; GADM 4.1. Diolah penulis (2026).</div>", unsafe_allow_html=True)
