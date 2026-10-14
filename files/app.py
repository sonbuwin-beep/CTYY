from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------
# Cau hinh chung
# ----------------------------------------------------------------------------
TEAL, ORANGE, GREY, LGREY, INK = "#176B73", "#D9702A", "#8D979F", "#D7DCE0", "#1B2630"
DATA_FILE = Path(__file__).parent / "Courses_of_Interest_Processed.xlsx"

COURSE_NAMES = {
    "Khóa A": "IT Fullstack", "Khóa B": "IT Part time", "Khóa C": "Nhật Bản (Fukuoka)",
    "Khóa D": "TechX Certificate", "Khóa E": "PCIC", "Khóa F": "Data Analytics", "Khóa G": "AI",
}
WEEKDAYS = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "CN"]
SENT_NAMES = {"tich_cuc": "Tích cực", "tieu_cuc": "Tiêu cực", "unknown": "Unknown"}
VAR_NAMES = {
    "tu_van_vien": "Tư vấn viên", "signup_month": "Tháng đăng ký", "quan_tam_khoa_hoc": "Khóa quan tâm",
    "MKT_xu_ly_data": "Mức chăm sóc", "signup_hour": "Giờ đăng ký", "signup_weekday": "Thứ đăng ký",
    "days_to_consult": "Số ngày đến tư vấn", "sentimatic": "Sentiment", "nha_mang": "Nhà mạng",
    "has_consult": "Đã chuyển tư vấn", "signup_weekend": "Cuối tuần", "loi_ten_rong": "Thiếu tên",
}
CAT_COLS = ["MKT_xu_ly_data", "nha_mang", "quan_tam_khoa_hoc", "sentimatic", "tu_van_vien"]
NUM_COLS = ["loi_ten_rong", "has_consult", "days_to_consult", "signup_hour",
            "signup_weekday", "signup_month", "signup_weekend"]
SMALL_N = 30      # duoi muc nay coi la mau qua nho
SOLID_N = 100     # tu muc nay tro len coi la dang tin

st.set_page_config(page_title="TechX Academy | Phân tích và dự báo chốt sale",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
<style>
.stApp {{ background:#F4F1EA; color:{INK}; }}
.block-container {{ padding-top:2rem; max-width:1320px; }}
h1,h2,h3 {{ color:{INK}; font-weight:700; letter-spacing:-0.01em; }}
button[data-baseweb="tab"] {{ font-size:1.02rem; font-weight:600; color:#5D656C; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{TEAL}; }}
div[data-baseweb="tab-highlight"] {{ background-color:{ORANGE}; }}
.kpi {{ background:#fff; border:1px solid #D9D3C4; border-radius:14px; padding:16px 18px; }}
.kpi .v {{ font-size:2rem; font-weight:700; color:{TEAL}; line-height:1.1; }}
.kpi .l {{ font-size:0.9rem; color:#5D656C; margin-top:4px; }}
.note {{ background:{INK}; color:#F4F1EA; border-radius:14px; padding:18px 22px; }}
.note .h {{ color:#F2A65A; font-weight:700; font-size:0.85rem; letter-spacing:0.05em; margin-bottom:8px; }}
.note ul {{ margin:0; padding-left:1.1rem; }}
.note li {{ margin-bottom:8px; line-height:1.45; }}
.card {{ background:#fff; border:1px solid #D9D3C4; border-radius:14px; padding:16px 20px; height:100%; }}
.card .h {{ font-weight:700; color:{TEAL}; margin-bottom:6px; }}
.lead {{ font-size:1.05rem; color:#3b4650; }}
.scope {{ background:#fff; border:1px solid #D9D3C4; border-left:5px solid {ORANGE}; border-radius:10px;
          padding:10px 16px; margin:6px 0 14px 0; font-size:0.95rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Ham tien ich
# ----------------------------------------------------------------------------
def fi(n) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def fp(x, d=1) -> str:
    return f"{x:.{d}f}".replace(".", ",") + "%"


def fd(x, d=2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def kpi(label: str, value: str):
    st.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                unsafe_allow_html=True)


def note(bullets: list[str], title: str = "NHẬN ĐỊNH"):
    items = "".join(f"<li>{b}</li>" for b in bullets if b)
    st.markdown(f'<div class="note"><div class="h">{title}</div><ul>{items}</ul></div>', unsafe_allow_html=True)


def card(title: str, lines: list[str]):
    items = "".join(f"<li>{b}</li>" for b in lines if b)
    st.markdown(f'<div class="card"><div class="h">{title}</div><ul style="margin:0;padding-left:1.1rem">{items}</ul></div>',
                unsafe_allow_html=True)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def chart_with_note(fig, bullets: list[str], table: pd.DataFrame | None = None):
    c1, c2 = st.columns([2.1, 1])
    with c1:
        show(fig)
    with c2:
        note(bullets)
    if table is not None:
        with st.expander("Xem bảng số liệu của biểu đồ"):
            st.dataframe(table, hide_index=True)


def empty_msg(text="Không có dữ liệu với lựa chọn hiện tại. Hãy nới bộ lọc hoặc giảm ngưỡng."):
    st.info(text)


def slider_n(label, lo, hi, val, key, step=1):
    """Slider an toan: neu khoang chi co 1 gia tri thi khong ve slider."""
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        return hi
    return st.slider(label, min_value=lo, max_value=hi, value=int(min(max(val, lo), hi)), step=step, key=key)


def style(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#E6E2D6", lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)


def new_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    fig.patch.set_facecolor("white")
    return fig, ax


def hbar(labels, vals, colors, fmt, xlabel, title, xmax=None, w=9, h=None):
    h = h or max(3.2, 0.5 * len(labels) + 1.6)
    fig, ax = new_fig(w, h)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    xmax = xmax or (max(max(vals), 1e-9) * 1.3)
    for i, v in enumerate(vals):
        ax.text(v + xmax * 0.012, i, fmt(i, v), va="center", fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold", color="#5D656C")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=12)
    style(ax, "x")
    fig.tight_layout()
    return fig, ax


def colbar(labels, vals, ns, colors, title, tb, ylim=None, ylabel="Tỉ lệ chốt (%)"):
    fig, ax = new_fig(8, 4.8)
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ylim = ylim or max(max(vals) * 1.25, tb * 1.25, 10)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + ylim * 0.015, fp(v), ha="center", fontsize=11, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1.2))
        ax.text(i, ylim * 0.02, f"{fi(n)}\nkhách", ha="center", fontsize=9,
                color="white" if colors[i] != LGREY else INK)
    ax.axhline(tb, color=INK, ls="--", lw=1.6)
    ax.set_ylim(0, ylim)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", color="#5D656C")
    ax.set_xlabel("Nét đứt: trung bình dữ liệu đang chọn " + fp(tb), fontsize=9.5, color="#5D656C")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=12)
    style(ax, "y")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Du lieu
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Đang đọc dữ liệu...")
def load_data(source) -> pd.DataFrame:
    return pd.read_excel(source)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["chot"] = df["thoi_gian_chuyen_sang_chot_sale"].notna().astype(int)
    df["co_tu_van"] = df["thoi_gian_chuyen_sang_tu_van"].notna().astype(int)
    t0 = pd.to_datetime(df["ngay_tao"], errors="coerce")
    df["gio"], df["thu"] = t0.dt.hour, t0.dt.dayofweek
    df["thang"] = t0.dt.to_period("M").astype(str).where(t0.notna())
    df["khoa"] = df["quan_tam_khoa_hoc_1"].map(COURSE_NAMES)
    df["tvv"] = df["tu_van_vien"].fillna("Chưa xác định")
    df["sent"] = df["sentimatic"].map(SENT_NAMES)
    return df


def rate(s: pd.Series) -> float:
    return float(s.mean() * 100) if len(s) else 0.0


def minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0


def course_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.dropna(subset=["khoa"]).groupby("khoa").agg(n=("chot", "size"), chot=("chot", "sum"))
    g["rate"] = g.chot / g.n * 100
    return g


def tv_table(d: pd.DataFrame) -> pd.DataFrame:
    x = d[d.tvv != "Chưa xác định"]
    g = x.groupby("tvv").agg(n=("chot", "size"), chot=("chot", "sum"))
    g["rate"] = g.chot / g.n * 100
    return g


def small_note(d: pd.DataFrame) -> str:
    sm = d[d.n < SMALL_N]
    if sm.empty:
        return ""
    names = ", ".join(f"{i} ({fi(n)} khách)" for i, n in zip(sm.index, sm.n))
    return f"Mẫu quá nhỏ (dưới {SMALL_N} khách), chưa đủ tin cậy: {names}."


def rename_cols(d: pd.DataFrame, first: str) -> pd.DataFrame:
    return d.reset_index().rename(columns={d.index.name or "index": first, "n": "Số khách", "chot": "Số chốt",
                                           "rate": "Tỉ lệ chốt (%)", "score": "Điểm"}).round(3)


# ----------------------------------------------------------------------------
# Mo hinh du bao (giong notebook ML_DuDoan_ChotSale, chay tren toan bo du lieu)
# ----------------------------------------------------------------------------
def build_ml_frame(raw: pd.DataFrame):
    d = raw.copy()
    y = d["thoi_gian_chuyen_sang_chot_sale"].notna().astype(int)
    t0 = pd.to_datetime(d["ngay_tao"], errors="coerce")
    t1 = pd.to_datetime(d["thoi_gian_chuyen_sang_tu_van"], errors="coerce")
    d["signup_hour"], d["signup_weekday"], d["signup_month"] = t0.dt.hour, t0.dt.dayofweek, t0.dt.month
    d["signup_weekend"] = d["signup_weekday"].isin([5, 6]).astype(int)
    d["has_consult"] = t1.notna().astype(int)
    d["days_to_consult"] = (t1 - t0).dt.total_seconds() / 86400
    d.loc[d["days_to_consult"] < 0, "days_to_consult"] = np.nan
    X = d[CAT_COLS + NUM_COLS].copy()
    X[CAT_COLS] = X[CAT_COLS].fillna("Chua ro").astype(str)
    return X, y


@st.cache_resource(show_spinner="Đang huấn luyện mô hình Random Forest (khoảng 20-40 giây)...")
def train_model(_raw: pd.DataFrame, data_key: tuple):
    X, y = build_ml_frame(_raw)
    X_dev, X_ho, y_dev, y_ho = train_test_split(X, y, test_size=0.05, stratify=y, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_dev, y_dev, test_size=15 / 95, stratify=y_dev, random_state=42)
    corr = X_tr[NUM_COLS].fillna(X_tr[NUM_COLS].median()).assign(chot=y_tr).corr()
    pre = ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median"))]), NUM_COLS),
        ("category", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                               ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=25,
                                                        drop="first", sparse_output=False))]), CAT_COLS),
    ])
    A_tr = pre.fit_transform(X_tr)
    model = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(A_tr, y_tr)
    proba = {"Train 80%": (y_tr.values, model.predict_proba(A_tr)[:, 1]),
             "Test 15%": (y_te.values, model.predict_proba(pre.transform(X_te))[:, 1]),
             "Holdout 5%": (y_ho.values, model.predict_proba(pre.transform(X_ho))[:, 1])}
    names = pre.get_feature_names_out()

    def source(n):
        for c in CAT_COLS:
            if n.startswith(f"category__{c}_"):
                return c
        return n.replace("numeric__", "")

    imp = pd.Series(model.feature_importances_, index=names).groupby(source).sum().sort_values(ascending=False)
    imp.index = [VAR_NAMES.get(i, i) for i in imp.index]
    ho_idx = X_ho.index
    meta = pd.DataFrame({
        "STT": _raw.loc[ho_idx, "STT"].values,
        "Tư vấn viên": _raw.loc[ho_idx, "tu_van_vien"].fillna("Chưa xác định").values,
        "Khóa": _raw.loc[ho_idx, "quan_tam_khoa_hoc_1"].map(COURSE_NAMES).values,
    })
    return {"model": model, "pre": pre, "corr": corr, "imp": imp, "proba": proba, "meta": meta,
            "sizes": (len(X_tr), len(X_te), len(X_ho)), "pos": (int(y_tr.sum()), int(y_te.sum()), int(y_ho.sum()))}


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp_ = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n = len(y)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp_) if (tp + fp_) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": float(roc_auc_score(y, p)),
            "cm": np.array([[tn, fp_], [fn, tp]])}


def fig_cm(cm, title):
    fig, ax = new_fig(5.4, 4.4)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Không chốt", "Chốt"], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Không chốt", "Chốt"], fontsize=11)
    ax.set_xlabel("Dự báo", fontsize=11, fontweight="bold", color="#5D656C")
    ax.set_ylabel("Thực tế", fontsize=11, fontweight="bold", color="#5D656C")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, fi(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    return fig


def fig_corr(corr):
    names = [VAR_NAMES.get(c, "Đã chốt" if c == "chot" else c) for c in corr.columns]
    fig, ax = new_fig(7.5, 6)
    mat = corr.values
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                    color="white" if abs(mat[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Tương quan giữa các biến số và việc chốt (tập Train)", fontsize=13, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Nap du lieu va bo loc
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Nguồn dữ liệu")
    up = st.file_uploader("Tải file Excel khác (tùy chọn)", type=["xlsx"], key="uploader")
    st.caption("Mặc định dùng file Courses_of_Interest_Processed.xlsx cùng thư mục với app.")

source = up if up is not None else (DATA_FILE if DATA_FILE.exists() else None)
if source is None:
    st.error("Không tìm thấy file Courses_of_Interest_Processed.xlsx. Hãy đặt file cạnh app.py hoặc tải lên ở thanh bên trái.")
    st.stop()

raw = load_data(source)
need = {"thoi_gian_chuyen_sang_chot_sale", "ngay_tao", "quan_tam_khoa_hoc_1", "tu_van_vien", "sentimatic",
        "nha_mang", "MKT_xu_ly_data", "quan_tam_khoa_hoc", "thoi_gian_chuyen_sang_tu_van", "STT"}
if not need.issubset(raw.columns):
    st.error("File thiếu cột: " + ", ".join(sorted(need - set(raw.columns))))
    st.stop()

df_all = prepare(raw)
TOTAL_ALL, WON_ALL = len(df_all), int(df_all.chot.sum())
TB_ALL = rate(df_all.chot)

months_all = sorted(df_all.thang.dropna().unique())
nets_all = sorted(df_all.nha_mang.dropna().unique())
with st.sidebar:
    st.markdown("### Bộ lọc dữ liệu")
    st.caption("Áp dụng cho tab 1, 3, 4, 5. Tab 2 và tab 6 luôn dùng toàn bộ dữ liệu.")
    f_courses = st.multiselect("Khóa học", list(COURSE_NAMES.values()), default=list(COURSE_NAMES.values()), key="f_courses")
    f_unknown = st.checkbox("Gồm khách chưa xác định khóa", value=True, key="f_unknown")
    if len(months_all) > 1:
        f_m = st.select_slider("Khoảng tháng đăng ký", options=months_all, value=(months_all[0], months_all[-1]), key="f_months")
    else:
        f_m = (months_all[0], months_all[0])
    f_nets = st.multiselect("Nhà mạng", nets_all, default=nets_all, key="f_nets")

full_range = (f_m[0], f_m[1]) == (months_all[0], months_all[-1])
tcmp = df_all.thang.fillna("0000-00")
in_range = (tcmp >= f_m[0]) & (tcmp <= f_m[1])
if full_range:
    in_range = in_range | df_all.thang.isna()
mask = (df_all.khoa.isin(f_courses) | (df_all.khoa.isna() & f_unknown)) & in_range & df_all.nha_mang.isin(f_nets)
df = df_all[mask]
if df.empty:
    st.warning("Bộ lọc hiện tại không còn khách nào. Hãy chọn lại bộ lọc ở thanh bên trái.")
    st.stop()

TOTAL, WON = len(df), int(df.chot.sum())
TB = rate(df.chot)
CONSULT = int(df.co_tu_van.sum())
WON_NC = int(((df.chot == 1) & (df.co_tu_van == 0)).sum())
FILTERED = TOTAL != TOTAL_ALL
SCOPE = (f"Đang xem {fi(TOTAL)}/{fi(TOTAL_ALL)} khách ({fp(TOTAL / TOTAL_ALL * 100)}) theo bộ lọc; "
         f"trung bình tỉ lệ chốt của dữ liệu đang chọn là {fp(TB)}." if FILTERED else
         f"Đang xem toàn bộ {fi(TOTAL)} khách; trung bình tỉ lệ chốt là {fp(TB)}.")


def scope_banner():
    st.markdown(f'<div class="scope">{SCOPE}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Giao dien
# ----------------------------------------------------------------------------
st.markdown("## TechX Academy: phân tích hiệu quả tư vấn và dự báo khách chốt sale")
st.markdown('<p class="lead">Báo cáo đi theo trình tự: tổng quan, chất lượng dữ liệu, ba trọng tâm phân tích '
            '(khóa học, tư vấn viên, khách dễ chốt), mô hình dự báo, rồi kết luận. Mỗi biểu đồ có bộ điều khiển riêng; '
            'nhận định bên cạnh tự cập nhật theo lựa chọn.</p>', unsafe_allow_html=True)

tabs = st.tabs(["1. Tổng quan", "2. Chất lượng dữ liệu", "3. Khóa học", "4. Tư vấn viên",
                "5. Khách dễ chốt", "6. Dự báo ML", "7. Kết luận"])

# ============================================================================
# 1. Tong quan
# ============================================================================
with tabs[0]:
    st.markdown("### Bối cảnh, vấn đề và mục tiêu")
    c1, c2 = st.columns(2)
    with c1:
        card("Bối cảnh", ["TechX Academy: trường học công nghệ tại Việt Nam, thành lập năm 2015.",
                          "Sản phẩm: IT Fullstack, IT Bachelor's Program, Lập trình viên Nhật Bản (Fukuoka), "
                          "IT Developer Part time, Lập trình Web, Data Analytics Foundation.",
                          f"Dữ liệu phân tích: {fi(TOTAL_ALL)} khách đăng ký tư vấn."])
    with c2:
        card("Vấn đề và mục tiêu", [
            "Vấn đề: các khóa học cần cải thiện để đáp ứng nhu cầu thị trường và xu hướng hiện nay, "
            "nhằm thu hút và chuyển đổi nhiều học viên hơn.",
            "Mục tiêu: đánh giá hiệu quả kinh doanh theo khóa học, tư vấn viên và nguồn khách, rồi dự báo khách có chốt sale hay không."])
    st.markdown("### Chỉ số chính")
    scope_banner()
    k = st.columns(5)
    with k[0]: kpi("Khách đăng ký", fi(TOTAL))
    with k[1]: kpi("Khách chốt sale", fi(WON))
    with k[2]: kpi("Tỉ lệ chốt", fp(TB))
    with k[3]: kpi("Có bước tư vấn", fp(CONSULT / TOTAL * 100))
    with k[4]: kpi("Chốt không qua tư vấn", fp(WON_NC / WON * 100) if WON else "0%")
    st.markdown("### Phễu bán hàng")
    fmode = st.radio("Hiển thị", ["Số khách", "% so với số đăng ký"], horizontal=True, key="funnel_mode")
    fvals = [TOTAL, CONSULT, WON]
    flabs = ["Đăng ký", "Chuyển sang tư vấn", "Chốt sale"]
    if fmode == "Số khách":
        fig, _ = hbar(flabs, fvals, [GREY, LGREY, TEAL], lambda i, v: fi(v), "Số khách", "Phễu bán hàng: số khách ở từng bước", h=3.4)
    else:
        pv = [v / TOTAL * 100 for v in fvals]
        fig, _ = hbar(flabs, pv, [GREY, LGREY, TEAL], lambda i, v: f"{fp(v)} ({fi(fvals[i])})", "% so với số đăng ký",
                      "Phễu bán hàng: % so với số đăng ký", xmax=130, h=3.4)
    bl = [f"{fi(TOTAL)} khách đăng ký, {fi(CONSULT)} chuyển sang tư vấn, {fi(WON)} chốt."]
    if WON:
        share = WON_NC / WON * 100
        if WON > CONSULT:
            bl.append("Số chốt nhiều hơn số khách qua bước tư vấn.")
        bl.append(f"{fp(share, 0)} khách chốt không qua bước tư vấn" +
                  (": có ít nhất hai luồng bán hàng." if share >= 50 else "; phần lớn khách chốt đều qua tư vấn."))
    chart_with_note(fig, bl, pd.DataFrame({"Bước": flabs, "Số khách": fvals,
                                           "% so với đăng ký": [round(v / TOTAL * 100, 1) for v in fvals]}))

# ============================================================================
# 2. Chat luong du lieu (toan bo du lieu)
# ============================================================================
with tabs[1]:
    st.markdown("### Dữ liệu có lỗi gì, xử lý ra sao")
    st.markdown(f'<div class="scope">Tab này luôn dùng toàn bộ {fi(TOTAL_ALL)} dòng, không áp dụng bộ lọc.</div>', unsafe_allow_html=True)
    n_ai = int((df_all.quan_tam_khoa_hoc == "AI").sum())
    n_la = int(df_all.khoa_hoc_la.sum()) if "khoa_hoc_la" in df_all else 0
    items = [
        ("Thiếu tên khách / tư vấn viên", int(df_all.loi_ten_rong.sum()) if "loi_ten_rong" in df_all else 0, "Gán nhãn 'Chưa xác định'"),
        ("Ngày tư vấn → chốt bị âm", int(df_all.loi_ngay_am.sum()) if "loi_ngay_am" in df_all else 0, "Giữ nguyên, loại khỏi phép tính thời gian"),
        ("Tên khóa học lạ hoặc mới", n_ai + n_la, f"{fi(n_ai)} dòng AI gán mã Khóa G; {fi(n_la)} dòng còn lại chờ nghiệp vụ xác nhận"),
        ("Mã khóa gán lệch", int(df_all.quan_tam_khoa_hoc_1_goc.notna().sum()) if "quan_tam_khoa_hoc_1_goc" in df_all else 0, "Sửa theo câu trả lời gốc, lưu mã cũ"),
        ("Nhà mạng không xác định", int((df_all.nha_mang == "Không xác định").sum()), "Thay bằng 'Không xác định'"),
    ]
    c1, c2 = st.columns(2)
    with c1:
        dq_mode = st.radio("Hiển thị theo", ["Số dòng", "% tổng số dòng"], horizontal=True, key="dq_mode")
    with c2:
        dq_sort = st.radio("Sắp xếp", ["Giảm dần theo số dòng", "Theo thứ tự quy trình"], horizontal=True, key="dq_sort")
    its = sorted(items, key=lambda x: -x[1]) if dq_sort.startswith("Giảm") else list(items)
    dlabs = [a for a, _, _ in its]
    dvals = [b for _, b, _ in its]
    dcols = [ORANGE if v == max(dvals) else GREY for v in dvals]
    if dq_mode == "Số dòng":
        fig, _ = hbar(dlabs, dvals, dcols, lambda i, v: f"{fi(v)} dòng", "Số dòng lỗi", "Số dòng lỗi theo nhóm",
                      xmax=max(max(dvals), 1) * 1.3, h=3.8)
    else:
        pv = [v / TOTAL_ALL * 100 for v in dvals]
        fig, _ = hbar(dlabs, pv, dcols, lambda i, v: fp(v, 2), "% tổng số dòng", "Tỉ lệ dòng lỗi theo nhóm",
                      xmax=max(max(pv), 0.1) * 1.3, h=3.8)
    top_item = max(items, key=lambda x: x[1])
    chart_with_note(fig, [
        f"Lỗi nhiều nhất: {top_item[0].lower()} với {fi(top_item[1])} dòng ({fp(top_item[1] / TOTAL_ALL * 100)} tổng số dòng).",
        f"Tổng cộng {fi(sum(v for _, v, _ in items))} dòng lỗi thuộc {sum(1 for _, v, _ in items if v)} nhóm.",
        "Không xóa dòng nào: lỗi được gán nhãn hoặc sửa, giá trị gốc giữ ở cột _goc."],
        pd.DataFrame({"Lỗi": [a for a, _, _ in items], "Số dòng": [b for _, b, _ in items], "Cách xử lý": [c for _, _, c in items]}))
    with st.expander("Quy đổi mã khóa học sang tên khóa học"):
        st.dataframe(pd.DataFrame({"Mã trong dữ liệu": list(COURSE_NAMES), "Tên khóa học": list(COURSE_NAMES.values())}), hide_index=True)
    with st.expander("Xem dữ liệu gốc (chọn số dòng)"):
        nrow = slider_n("Số dòng hiển thị", 5, 200, 20, "dq_rows", 5)
        st.dataframe(raw.head(nrow))

# ============================================================================
# 3. Khoa hoc
# ============================================================================
with tabs[2]:
    st.markdown("### Trọng tâm 1: Khóa học nào tiềm năng nhất?")
    st.caption("Đăng ký nhiều → Chốt nhiều → Điểm KPI")
    scope_banner()
    gc = course_table(df)
    if gc.empty:
        empty_msg("Không có khách nào đã xác định khóa trong bộ lọc hiện tại.")
    else:
        t1, t2, t3 = st.tabs(["Đăng ký nhiều", "Chốt nhiều", "Điểm tiềm năng"])
        with t1:
            a, b = st.columns(2)
            with a:
                top_n = slider_n("Số khóa hiển thị (Top)", 1, len(gc), len(gc), "c_demand_n")
            with b:
                dmode = st.radio("Hiển thị", ["Số khách", "% tổng số khách"], horizontal=True, key="c_demand_mode")
            d = gc.sort_values("n", ascending=False).head(top_n)
            tot = gc.n.sum()
            cols = [TEAL if i == 0 else GREY for i in range(len(d))]
            if dmode == "Số khách":
                fig, _ = hbar(list(d.index), d.n.values, cols, lambda i, v: f"{fi(v)} ({fp(v / tot * 100)})", "Số khách đăng ký",
                              f"Số khách đăng ký theo khóa học (Top {len(d)})")
            else:
                fig, _ = hbar(list(d.index), d.n.values / tot * 100, cols, lambda i, v: f"{fp(v)} ({fi(d.n.iloc[i])})",
                              "% tổng số khách đã xác định khóa", f"Cơ cấu đăng ký theo khóa học (Top {len(d)})", xmax=130)
            bl = [f"{d.index[0]}: {fi(d.n.iloc[0])} khách, chiếm {fp(d.n.iloc[0] / tot * 100)} số khách đã xác định khóa."]
            if len(d) >= 2:
                bl.append(f"Gấp {fd(d.n.iloc[0] / d.n.iloc[1], 1)} lần khóa đứng thứ hai ({d.index[1]}, {fi(d.n.iloc[1])} khách).")
            bl.append(f"{len(d)} khóa đang hiển thị gồm {fi(d.n.sum())} khách ({fp(d.n.sum() / tot * 100)} tổng); "
                      f"tổng cộng có {len(gc)} khóa và {fi(tot)} khách đã xác định khóa.")
            chart_with_note(fig, bl, rename_cols(d, "Khóa"))
        with t2:
            a, b = st.columns(2)
            with a:
                min_n = slider_n("Số khách tối thiểu của mỗi khóa", 0, int(gc.n.max()), 0, "c_rate_min", 10)
            with b:
                sort_by = st.selectbox("Sắp xếp theo", ["Tỉ lệ chốt", "Số khách"], index=0, key="c_rate_sort")
            d = gc[gc.n >= min_n].sort_values("rate" if sort_by == "Tỉ lệ chốt" else "n", ascending=False)
            if d.empty:
                empty_msg()
            else:
                cols = [TEAL if r >= TB else (ORANGE if r < TB * 0.5 else GREY) for r in d.rate]
                fig, ax = hbar(list(d.index), d.rate.values, cols, lambda i, v: f"{fp(v)}  ({fi(d.n.iloc[i])} khách)",
                               "Tỉ lệ chốt sale (%)", "Tỉ lệ chốt sale theo khóa học (nét đứt: trung bình đang chọn)",
                               xmax=max(d.rate.max(), TB) * 1.5)
                ax.axvline(TB, color=INK, ls="--", lw=1.6)
                bi, wi = d.rate.idxmax(), d.rate.idxmin()
                big = d.n.idxmax()
                bl = [f"Cao nhất: {bi} {fp(d.rate[bi])} ({fi(d.n[bi])} khách); thấp nhất: {wi} {fp(d.rate[wi])} ({fi(d.n[wi])} khách).",
                      f"{int((d.rate >= TB).sum())}/{len(d)} khóa có tỉ lệ chốt từ mức trung bình đang chọn ({fp(TB)}) trở lên.",
                      f"Khóa đông khách nhất ({big}, {fi(d.n[big])} khách) chốt {fp(d.rate[big])}, "
                      f"{'cao hơn' if d.rate[big] >= TB else 'thấp hơn'} trung bình {fd(abs(d.rate[big] - TB), 1)} điểm phần trăm.",
                      small_note(d)]
                chart_with_note(fig, bl, rename_cols(d, "Khóa"))
        with t3:
            a, b, c = st.columns(3)
            with a:
                w = st.slider("Trọng số lượng đăng ký (%)", min_value=0, max_value=100, value=50, step=10, key="c_score_w")
            with b:
                min_n3 = slider_n("Số khách tối thiểu", 0, int(gc.n.max()), 0, "c_score_min", 10)
            with c:
                st.caption(f"Trọng số hiệu quả chốt tự động = {100 - w}%")
            d = gc[gc.n >= min_n3].copy()
            if d.empty:
                empty_msg()
            else:
                d["score"] = (w / 100) * minmax(d.n) + (1 - w / 100) * minmax(d.rate)
                d = d.sort_values("score", ascending=False)
                n_elig = len(d)
                top3 = slider_n("Số khóa hiển thị (Top)", 1, n_elig, n_elig, "c_score_n")
                d = d.head(top3)
                cols = [TEAL if i == 0 else (ORANGE if s < 0.15 else GREY) for i, s in enumerate(d.score)]
                fig, _ = hbar(list(d.index), d.score.values, cols, lambda i, v: fd(v, 3), "Điểm tiềm năng (0 đến 1)",
                              f"Điểm tiềm năng ({w}% đăng ký + {100 - w}% tỉ lệ chốt)", xmax=1.15)
                bl = [f"{d.index[0]} dẫn đầu với {fd(d.score.iloc[0], 3)} điểm ({fi(d.n.iloc[0])} khách, chốt {fp(d.rate.iloc[0])})."]
                if len(d) >= 2:
                    bl.append(f"{d.index[1]} đứng thứ hai: {fd(d.score.iloc[1], 3)} điểm, {fi(d.n.iloc[1])} khách.")
                bl.append(f"Điểm chuẩn hóa 0 đến 1 trong số {n_elig} khóa đủ điều kiện; đổi trọng số để xem thứ hạng thay đổi.")
                bl.append(small_note(d))
                chart_with_note(fig, bl, rename_cols(d, "Khóa"))

# ============================================================================
# 4. Tu van vien
# ============================================================================
with tabs[3]:
    st.markdown("### Trọng tâm 2: Tư vấn viên nào hiệu quả?")
    st.caption("Nhận nhiều khách → Chốt hiệu quả")
    scope_banner()
    gt = tv_table(df)
    if gt.empty:
        empty_msg("Không có khách nào có tên tư vấn viên trong bộ lọc hiện tại.")
    else:
        t1, t2, t3, t4, t5 = st.tabs(["Nhận nhiều khách", "Độ tập trung lead", "Xếp hạng chốt",
                                      "So sánh tư vấn viên", "Bảng xếp hạng"])
        with t1:
            a, b = st.columns(2)
            with a:
                topn = slider_n("Số tư vấn viên hiển thị (Top)", 3, min(30, len(gt)), min(10, len(gt)), "tv_vol_n")
            with b:
                sortb = st.selectbox("Xếp theo", ["Số khách nhận", "Tỉ lệ chốt"], index=0, key="tv_vol_sort")
            skey = "n" if sortb == "Số khách nhận" else "rate"
            d = gt.sort_values(skey, ascending=False).head(topn)
            cols = [TEAL if r >= TB else ORANGE for r in d.rate]
            fig, _ = hbar(list(d.index), d.n.values, cols, lambda i, v: f"{fi(v)} · {fp(d.rate.iloc[i])}",
                          "Số khách nhận (xanh/cam: tỉ lệ chốt trên/dưới trung bình đang chọn)",
                          f"Top {len(d)} tư vấn viên theo {sortb.lower()}", xmax=d.n.max() * 1.45)
            lo, hi = d.rate.idxmin(), d.rate.idxmax()
            bl = [f"{d.index[0]} đứng đầu theo {sortb.lower()}: {fi(d.n.iloc[0])} khách, chốt {fp(d.rate.iloc[0])}."]
            if lo != hi:
                bl.append(f"Trong nhóm này: {hi} chốt cao nhất ({fp(d.rate[hi])} trên {fi(d.n[hi])} khách), "
                          f"{lo} chốt thấp nhất ({fp(d.rate[lo])} trên {fi(d.n[lo])} khách).")
            bl.append(f"{int((d.rate >= TB).sum())}/{len(d)} người chốt từ mức trung bình đang chọn ({fp(TB)}) trở lên.")
            bl.append(f"Nhóm hiển thị nhận {fi(d.n.sum())} khách ({fp(d.n.sum() / gt.n.sum() * 100)} tổng lead có tư vấn viên).")
            chart_with_note(fig, bl, rename_cols(d, "Tư vấn viên"))
        with t2:
            thr = st.slider("Ngưỡng % lead cộng dồn", min_value=50, max_value=95, value=80, step=5, key="tv_pareto")
            s = gt.sort_values("n", ascending=False)
            cum = (s.n.cumsum() / s.n.sum() * 100).values
            kk = int((cum < thr).sum() + 1)
            fig, ax = new_fig(8, 4.8)
            ax.plot(range(1, len(s) + 1), cum, color=TEAL, lw=2.5, marker="o", ms=3.5)
            ax.axhline(thr, color=GREY, ls="--", lw=1.5)
            ax.axvline(kk, color=ORANGE, ls="--", lw=1.5)
            ax.plot([kk], [cum[kk - 1]], "o", color=ORANGE, ms=9, zorder=5)
            ax.set_ylim(0, 105)
            ax.set_xlabel("Số tư vấn viên (xếp theo số khách giảm dần)", fontsize=11, fontweight="bold", color="#5D656C")
            ax.set_ylabel("% lead cộng dồn", fontsize=11, fontweight="bold", color="#5D656C")
            ax.set_title("Độ tập trung lead (Pareto)", fontsize=14, fontweight="bold", loc="left", pad=12)
            style(ax, "y")
            fig.tight_layout()
            ratio = kk / len(s) * 100
            chart_with_note(fig, [f"{kk}/{len(s)} tư vấn viên ({fp(ratio, 0)}) giữ {thr}% tổng lead.",
                                  f"Người nhận nhiều nhất: {s.index[0]} ({fi(s.n.iloc[0])} khách, {fp(s.n.iloc[0] / s.n.sum() * 100)} tổng lead).",
                                  ("Lead tập trung vào nhóm nhỏ: rủi ro khi vài người nghỉ." if ratio <= 40
                                   else "Lead được chia khá đều giữa các tư vấn viên.")],
                            pd.DataFrame({"Số tư vấn viên": range(1, len(s) + 1), "% lead cộng dồn": np.round(cum, 1)}))
        with t3:
            a, b, c = st.columns(3)
            with a:
                which = st.radio("Xem nhóm", ["Chốt cao nhất", "Chốt thấp nhất"], horizontal=True, key="tv_rank_which")
            with b:
                min_n = slider_n("Số khách tối thiểu để xếp hạng", 5, 300, 10, "tv_rank_min", 5)
            with c:
                topk = slider_n("Số người hiển thị", 3, 30, 10, "tv_rank_n")
            top = which == "Chốt cao nhất"
            d = gt[gt.n >= min_n].sort_values("rate", ascending=not top).head(topk)
            if d.empty:
                empty_msg()
            else:
                labels = [f"{a_} ({fi(b_)})" for a_, b_ in zip(d.index, d.n)]
                if top:
                    cols = [TEAL if n >= SOLID_N else LGREY for n in d.n]
                    title = f"Top {len(d)} tỉ lệ chốt cao nhất (xám nhạt: dưới {SOLID_N} khách)"
                else:
                    cols = [ORANGE if r < 1 else GREY for r in d.rate]
                    title = f"{len(d)} tỉ lệ chốt thấp nhất (nét đứt: trung bình đang chọn)"
                fig, ax = hbar(labels, d.rate.values, cols, lambda i, v: fp(v), "Tỉ lệ chốt (%), trong ngoặc là số khách",
                               title, xmax=max(d.rate.max() * 1.25, TB * 1.3, 10))
                if not top:
                    ax.axvline(TB, color=INK, ls="--", lw=1.6)
                solid = d[d.n >= SOLID_N]
                if top:
                    bl = [f"{len(solid)}/{len(d)} người trong nhóm có từ {SOLID_N} khách trở lên (kết quả đáng tin)."]
                    if len(solid):
                        bl.append("Đáng tin nhất: " + ", ".join(f"{a_} ({fi(b_)} khách, {fp(c_)})" for a_, b_, c_ in
                                                                  zip(solid.index[:3], solid.n[:3], solid.rate[:3])) + ".")
                    sm = d[d.n < SMALL_N]
                    if len(sm):
                        bl.append("Mẫu quá nhỏ: " + ", ".join(f"{a_} ({fi(b_)} khách)" for a_, b_ in zip(sm.index, sm.n)) + ".")
                else:
                    zero = int((d.rate < 1).sum())
                    first = (f"{zero} người chốt dưới 1% dù có từ {min_n} khách trở lên." if zero
                             else f"Thấp nhất: {d.index[0]} ({fp(d.rate.iloc[0])} trên {fi(d.n.iloc[0])} khách).")
                    bl = [first,
                          f"{len(d)} người này giữ {fi(d.n.sum())} khách ({fp(d.n.sum() / gt.n.sum() * 100)} tổng lead có tư vấn viên).",
                          f"Thấp hơn trung bình đang chọn ({fp(TB)}): {int((d.rate < TB).sum())}/{len(d)} người."]
                chart_with_note(fig, bl, rename_cols(d, "Tư vấn viên"))
        with t4:
            order = list(gt.sort_values("n", ascending=False).index)
            default = list(gt[gt.n >= SOLID_N].sort_values("rate", ascending=False).index[:4]) or order[:4]
            pick = st.multiselect("Chọn các tư vấn viên cần so sánh (tối đa 12)", order, default=default,
                                  max_selections=12, key="tv_cmp")
            if len(pick) < 2:
                empty_msg("Hãy chọn ít nhất 2 tư vấn viên để so sánh.")
            else:
                d = gt.loc[pick].sort_values("rate", ascending=False)
                c1, c2 = st.columns(2)
                with c1:
                    fig, _ = hbar(list(d.index), d.n.values, [GREY] * len(d), lambda i, v: fi(v), "Số khách nhận",
                                  "Số khách nhận", xmax=d.n.max() * 1.25)
                    show(fig)
                with c2:
                    fig, ax = hbar(list(d.index), d.rate.values, [TEAL if r >= TB else ORANGE for r in d.rate],
                                   lambda i, v: fp(v), "Tỉ lệ chốt (%)", "Tỉ lệ chốt (nét đứt: trung bình đang chọn)",
                                   xmax=max(d.rate.max() * 1.25, TB * 1.3, 10))
                    ax.axvline(TB, color=INK, ls="--", lw=1.6)
                    show(fig)
                hi, lo = d.rate.idxmax(), d.rate.idxmin()
                note([f"Cao nhất: {hi} {fp(d.rate[hi])} ({fi(d.n[hi])} khách); thấp nhất: {lo} {fp(d.rate[lo])} ({fi(d.n[lo])} khách).",
                      f"Chênh nhau {fd(d.rate[hi] - d.rate[lo], 1)} điểm phần trăm" +
                      (f", gấp {fd(d.rate[hi] / d.rate[lo], 1)} lần." if d.rate[lo] > 0 else "; người thấp nhất chưa chốt được khách nào."),
                      f"Nhận nhiều khách nhất trong nhóm chọn: {d.n.idxmax()} ({fi(d.n.max())} khách, chốt {fp(d.rate[d.n.idxmax()])}).",
                      small_note(d)])
                with st.expander("Xem bảng số liệu của biểu đồ"):
                    st.dataframe(rename_cols(d, "Tư vấn viên"), hide_index=True)
        with t5:
            q = st.text_input("Tìm theo tên tư vấn viên", value="", key="tv_search")
            tbl = rename_cols(gt, "Tư vấn viên")
            if q.strip():
                tbl = tbl[tbl["Tư vấn viên"].str.contains(q.strip(), case=False, na=False)]
            st.dataframe(tbl.sort_values("Số khách", ascending=False), hide_index=True, height=420)

# ============================================================================
# 5. Khach de chot
# ============================================================================
with tabs[4]:
    st.markdown("### Trọng tâm 3: Khách nào dễ chốt hơn?")
    st.caption("Thời điểm → Đặc điểm khách → Tốc độ chốt")
    scope_banner()
    t1, t2, t3, t4, t5, t6 = st.tabs(["Tháng", "Thứ trong tuần", "Giờ đăng ký", "Giờ hành chính",
                                      "Nhóm khách", "Tốc độ chốt"])
    with t1:
        min_lead = slider_n("Số lead tối thiểu của mỗi tháng", 0, 500, 50, "m_min", 10)
        m = df.groupby("thang").agg(n=("chot", "size"), r=("chot", "mean"))
        m = m[m.n >= min_lead]
        m["r"] *= 100
        if m.empty:
            empty_msg()
        else:
            fig, ax = new_fig(10, 5)
            x = np.arange(len(m))
            ax.bar(x, m.n, color=LGREY, width=0.62)
            ax.set_xticks(x)
            ax.set_xticklabels([f"T{int(s_[-2:])}/{s_[2:4]}" for s_ in m.index], fontsize=10)
            for i, v in enumerate(m.n):
                ax.text(i, m.n.max() * 0.02, fi(v), ha="center", fontsize=8.5, color="#5D656C")
            ax.set_ylim(0, m.n.max() * 1.15)
            ax.set_ylabel("Số lead", fontsize=11, fontweight="bold", color="#5D656C")
            ax2 = ax.twinx()
            ax2.plot(x, m.r, color=TEAL, lw=2.8, marker="o", ms=7)
            pk = int(np.argmax(m.r.values))
            ax2.plot([pk], [m.r.iloc[pk]], "o", color=ORANGE, ms=12, zorder=6)
            for i, v in enumerate(m.r):
                ax2.text(i, v + 3, fp(v), ha="center", fontsize=9.5, fontweight="bold", color=ORANGE if i == pk else TEAL)
            ax2.set_ylim(0, max(70, m.r.max() * 1.2))
            ax2.set_ylabel("Tỉ lệ chốt (%)", fontsize=11, fontweight="bold", color="#5D656C")
            ax.set_title("Số lead và tỉ lệ chốt theo tháng đăng ký", fontsize=14, fontweight="bold", loc="left", pad=12)
            ax.set_xlabel("Cột xám: số lead. Đường xanh: tỉ lệ chốt", fontsize=9.5, color="#5D656C")
            style(ax, "y")
            ax2.spines["top"].set_visible(False)
            fig.tight_layout()
            pkm, lom = m.r.idxmax(), m.r.idxmin()
            bl = [f"Đỉnh tháng {int(pkm[-2:])}/{pkm[2:4]}: {fp(m.r.max())} với {fi(m.n[pkm])} lead; "
                  f"thấp nhất tháng {int(lom[-2:])}/{lom[2:4]}: {fp(m.r.min())} ({fi(m.n[lom])} lead)."]
            if len(m) >= 2:
                a_, b_ = m.index[0], m.index[-1]
                bl.append(f"Từ tháng {int(a_[-2:])}/{a_[2:4]} ({fp(m.r.iloc[0])}) đến tháng {int(b_[-2:])}/{b_[2:4]} ({fp(m.r.iloc[-1])}): "
                          f"{'tăng' if m.r.iloc[-1] > m.r.iloc[0] else 'giảm'} {fd(abs(m.r.iloc[-1] - m.r.iloc[0]), 1)} điểm phần trăm.")
            bl.append(f"Đang hiển thị {len(m)} tháng có từ {min_lead} lead; cần ít nhất hai năm dữ liệu để kết luận chắc về tính mùa vụ.")
            chart_with_note(fig, bl, m.reset_index().rename(columns={"thang": "Tháng", "n": "Số lead", "r": "Tỉ lệ chốt (%)"}).round(1))
    with t2:
        wmode = st.radio("Biểu đồ hiển thị", ["Số lead", "Tỉ lệ chốt"], horizontal=True, key="w_mode")
        w = df.groupby("thu").agg(n=("chot", "size"), r=("chot", "mean")).reindex(range(7))
        w["r"] = w["r"] * 100
        w = w.fillna(0)
        fig, ax = new_fig(8, 4.8)
        x = np.arange(7)
        colv = w.n if wmode == "Số lead" else w.r
        best = int(colv.values.argmax())
        ax.bar(x, colv, color=[TEAL if i == best else (LGREY if i >= 5 else GREY) for i in range(7)], width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(WEEKDAYS, fontsize=11)
        top_v = max(colv.max(), 1)
        for i in range(7):
            ax.text(i, colv.iloc[i] + top_v * 0.015, fi(colv.iloc[i]) if wmode == "Số lead" else fp(colv.iloc[i]),
                    ha="center", fontsize=10, fontweight="bold")
            ax.text(i, top_v * 0.02, (fp(w.r.iloc[i]) if wmode == "Số lead" else f"{fi(w.n.iloc[i])} lead"),
                    ha="center", fontsize=9, color="white" if i < 5 else INK)
        if wmode == "Tỉ lệ chốt":
            ax.axhline(TB, color=INK, ls="--", lw=1.6)
        ax.set_ylim(0, top_v * 1.18)
        ax.set_ylabel("Số lead" if wmode == "Số lead" else "Tỉ lệ chốt (%)", fontsize=11, fontweight="bold", color="#5D656C")
        ax.set_title("Số lead theo thứ đăng ký" if wmode == "Số lead" else "Tỉ lệ chốt theo thứ đăng ký",
                     fontsize=14, fontweight="bold", loc="left", pad=12)
        style(ax, "y")
        fig.tight_layout()
        wk_lead, we_lead = w.n.iloc[:5].sum() / 5, w.n.iloc[5:].sum() / 2
        wk_rate, we_rate = rate(df[df.thu < 5].chot), rate(df[df.thu >= 5].chot)
        bl = [f"{WEEKDAYS[int(w.n.values.argmax())]} nhiều lead nhất: {fi(w.n.max())}; {WEEKDAYS[int(w.n.values.argmin())]} ít nhất: {fi(w.n.min())}.",
              f"Tỉ lệ chốt cao nhất: {WEEKDAYS[int(w.r.values.argmax())]} ({fp(w.r.max())}); thấp nhất: {WEEKDAYS[int(w.r.values.argmin())]} ({fp(w.r.min())}).",
              f"Cuối tuần: trung bình {fi(we_lead)} lead mỗi ngày, chốt {fp(we_rate)}; ngày thường: {fi(wk_lead)} lead mỗi ngày, chốt {fp(wk_rate)} "
              f"({'cuối tuần chốt cao hơn' if we_rate > wk_rate else 'ngày thường chốt cao hơn'} {fd(abs(we_rate - wk_rate), 1)} điểm phần trăm)."]
        chart_with_note(fig, bl, pd.DataFrame({"Thứ": WEEKDAYS, "Số lead": w.n.astype(int).values, "Tỉ lệ chốt (%)": w.r.round(1).values}))
    with t3:
        a, b = st.columns(2)
        with a:
            hr = st.slider("Khoảng giờ hiển thị", min_value=0, max_value=23, value=(0, 23), key="h_range")
        with b:
            min_h = slider_n("Số lead tối thiểu của mỗi giờ", 0, 500, 30, "h_min", 10)
        h = df[df.gio.between(hr[0], hr[1])].groupby("gio").agg(n=("chot", "size"), r=("chot", "mean"))
        h = h[h.n >= min_h]
        h["r"] *= 100
        if h.empty:
            empty_msg()
        else:
            fig, ax = new_fig(10, 5)
            x = np.arange(len(h))
            ax.bar(x, h.n, color=LGREY, width=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(i)}h" for i in h.index], fontsize=9.5)
            ax.set_ylim(0, h.n.max() * 1.15)
            ax.set_ylabel("Số lead", fontsize=11, fontweight="bold", color="#5D656C")
            ax2 = ax.twinx()
            ax2.plot(x, h.r, color=TEAL, lw=2.6, marker="o", ms=5)
            ax2.set_ylim(0, max(60, h.r.max() * 1.2))
            ax2.set_ylabel("Tỉ lệ chốt (%)", fontsize=11, fontweight="bold", color="#5D656C")
            ax.set_xlabel(f"Giờ đăng ký (giờ có từ {min_h} lead trở lên)", fontsize=9.5, color="#5D656C")
            ax.set_title("Số lead và tỉ lệ chốt theo giờ đăng ký", fontsize=14, fontweight="bold", loc="left", pad=12)
            style(ax, "y")
            ax2.spines["top"].set_visible(False)
            fig.tight_layout()
            bh, hh, lh = h.n.idxmax(), h.r.idxmax(), h.r.idxmin()
            bl = [f"Đông nhất lúc {int(bh)}h: {fi(h.n[bh])} lead, chốt {fp(h.r[bh])}.",
                  f"Tỉ lệ chốt cao nhất lúc {int(hh)}h ({fp(h.r[hh])}, {fi(h.n[hh])} lead); thấp nhất lúc {int(lh)}h ({fp(h.r[lh])}, {fi(h.n[lh])} lead).",
                  f"Trong {len(h)} giờ đang hiển thị, tỉ lệ chốt dao động từ {fp(h.r.min())} đến {fp(h.r.max())}; giờ ít lead cho tỉ lệ biến động mạnh vì mẫu nhỏ."]
            chart_with_note(fig, bl, h.reset_index().rename(columns={"gio": "Giờ", "n": "Số lead", "r": "Tỉ lệ chốt (%)"}).round(1))
    with t4:
        hs = st.slider("Khung giờ hành chính", min_value=0, max_value=23, value=(8, 17), key="hc_range")
        inside = df.gio.between(hs[0], hs[1])
        hvals = [rate(df[inside].chot), rate(df[~inside].chot)]
        hns = [int(inside.sum()), int((~inside).sum())]
        if min(hns) == 0:
            empty_msg("Khung giờ này bao trùm toàn bộ dữ liệu hoặc không có khách nào; hãy thu hẹp lại.")
        else:
            fig, ax = new_fig(6.5, 4.6)
            ax.bar([f"Trong khung\n({hs[0]}h đến {hs[1]}h)", "Ngoài khung"], hvals, color=[TEAL, GREY], width=0.5)
            for i, (v, n) in enumerate(zip(hvals, hns)):
                ax.text(i, v + max(hvals) * 0.02, f"{fp(v)} ({fi(n)} khách)", ha="center", fontsize=10.5, fontweight="bold")
            ax.axhline(TB, color=INK, ls="--", lw=1.4)
            ax.set_ylim(0, max(max(hvals) * 1.3, 10))
            ax.set_ylabel("Tỉ lệ chốt (%)", fontsize=11, fontweight="bold", color="#5D656C")
            ax.set_title("Trong khung giờ so với ngoài khung giờ", fontsize=14, fontweight="bold", loc="left", pad=12)
            style(ax, "y")
            fig.tight_layout()
            diff = hvals[0] - hvals[1]
            chart_with_note(fig, [f"Trong khung {hs[0]}h đến {hs[1]}h: {fp(hvals[0])} ({fi(hns[0])} khách); ngoài khung: {fp(hvals[1])} ({fi(hns[1])} khách).",
                                  f"Chênh {fd(abs(diff), 1)} điểm phần trăm: " +
                                  ("không có khác biệt đáng kể." if abs(diff) < 2 else f"{'trong' if diff > 0 else 'ngoài'} khung giờ chốt cao hơn."),
                                  ("Nhóm ít khách hơn có dưới 1.000 khách nên kết quả kém ổn định." if min(hns) < 1000 else "")],
                            pd.DataFrame({"Nhóm": ["Trong khung", "Ngoài khung"], "Số khách": hns, "Tỉ lệ chốt (%)": np.round(hvals, 1)}))
    with t5:
        a, b = st.columns(2)
        with a:
            var = st.selectbox("So sánh tỉ lệ chốt theo", ["Sentiment", "Mức chăm sóc", "Nhà mạng", "Khóa học"], index=0, key="g_var")
        with b:
            min_g = slider_n("Số khách tối thiểu của mỗi nhóm", 0, 1000, 30, "g_min", 10)
        colname = {"Sentiment": "sent", "Mức chăm sóc": "MKT_xu_ly_data", "Nhà mạng": "nha_mang", "Khóa học": "khoa"}[var]
        g = df.dropna(subset=[colname]).groupby(colname).agg(n=("chot", "size"), r=("chot", "mean"))
        g = g[g.n >= min_g]
        g["r"] *= 100
        g = g.sort_values("r", ascending=False)
        if g.empty:
            empty_msg()
        else:
            cols = [TEAL if r >= TB else (ORANGE if r < TB * 0.8 else GREY) for r in g.r]
            fig = colbar(list(g.index), list(g.r), list(g.n), cols, f"Tỉ lệ chốt theo {var.lower()}", TB)
            hi, lo = g.r.idxmax(), g.r.idxmin()
            bl = [f"Cao nhất: {hi} ({fp(g.r[hi])}, {fi(g.n[hi])} khách); thấp nhất: {lo} ({fp(g.r[lo])}, {fi(g.n[lo])} khách).",
                  f"Chênh giữa nhóm cao nhất và thấp nhất: {fd(g.r[hi] - g.r[lo], 1)} điểm phần trăm."]
            if var == "Sentiment" and {"Tích cực", "Tiêu cực"} <= set(g.index):
                d_ = g.r["Tích cực"] - g.r["Tiêu cực"]
                bl.append(f"Tích cực {fp(g.r['Tích cực'])} so với Tiêu cực {fp(g.r['Tiêu cực'])}: " +
                          ("gần bằng nhau, sentiment không phân biệt được khách chốt." if abs(d_) < 2 else
                           f"{'tích cực' if d_ > 0 else 'tiêu cực'} chốt cao hơn {fd(abs(d_), 1)} điểm phần trăm."))
            if var == "Nhà mạng":
                bl.append("Đây là nhà mạng SIM, không phải kênh marketing: cần bổ sung dữ liệu kênh để đánh giá nguồn khách.")
            sm = g[g.n < SMALL_N]
            if len(sm):
                bl.append("Mẫu quá nhỏ: " + ", ".join(f"{i} ({fi(n)} khách)" for i, n in zip(sm.index, sm.n)) + ".")
            chart_with_note(fig, bl, g.reset_index().rename(columns={colname: var, "n": "Số khách", "r": "Tỉ lệ chốt (%)"}).round(1))
    with t6:
        a, b, c = st.columns(3)
        with a:
            scheme = st.selectbox("Cách chia nhóm ngày", ["Chi tiết (≤1, 2-3, 4-7, 8-14, 15-30, >30)",
                                                          "Theo tuần (≤7, 8-14, 15-21, 22-30, >30)"], index=0, key="sp_scheme")
        with b:
            smode = st.radio("Hiển thị", ["% ca chốt", "Số ca"], horizontal=True, key="sp_mode")
        with c:
            slow = st.slider("Ngưỡng 'chốt chậm' (ngày)", min_value=5, max_value=60, value=15, step=1, key="sp_slow")
        dc = df[(df.chot == 1) & df.so_ngay_tu_van_den_chot.notna()]
        if dc.empty:
            empty_msg("Không có ca chốt nào có ngày hợp lệ trong bộ lọc hiện tại.")
        else:
            if scheme.startswith("Chi tiết"):
                edges, slabs = [-1, 1, 3, 7, 14, 30, 9999], ["≤1 ngày", "2-3 ngày", "4-7 ngày", "8-14 ngày", "15-30 ngày", ">30 ngày"]
            else:
                edges, slabs = [-1, 7, 14, 21, 30, 9999], ["≤7 ngày", "8-14 ngày", "15-21 ngày", "22-30 ngày", ">30 ngày"]
            cnt = pd.cut(dc.so_ngay_tu_van_den_chot, edges, labels=slabs).value_counts().reindex(slabs)
            svals = (cnt / cnt.sum() * 100).values if smode == "% ca chốt" else cnt.values
            fig, ax = new_fig(9, 4.8)
            ax.bar(slabs, svals, color=[TEAL if (e > slow) else GREY for e in edges[1:]], width=0.6)
            for i, v in enumerate(svals):
                ax.text(i, v + max(svals) * 0.02, fp(v) if smode == "% ca chốt" else fi(v), ha="center", fontsize=11, fontweight="bold")
            ax.set_ylim(0, max(svals) * 1.25)
            ax.set_ylabel(smode, fontsize=11, fontweight="bold", color="#5D656C")
            ax.set_xlabel(f"Số ngày từ tư vấn đến chốt (tính trên {fi(len(dc))} ca chốt có ngày hợp lệ; xanh: nhóm có cận trên vượt {slow} ngày)",
                          fontsize=9, color="#5D656C")
            ax.set_title("Thời gian từ tư vấn đến chốt", fontsize=14, fontweight="bold", loc="left", pad=12)
            style(ax, "y")
            fig.tight_layout()
            share_slow = float((dc.so_ngay_tu_van_den_chot > slow).mean() * 100)
            share_fast = float((dc.so_ngay_tu_van_den_chot <= 1).mean() * 100)
            chart_with_note(fig, [f"Chốt trong 1 ngày: {fp(share_fast)} số ca.",
                                  f"{fp(share_slow)} ca chốt mất hơn {slow} ngày; trung bình {fd(dc.so_ngay_tu_van_den_chot.mean(), 1)} ngày, "
                                  f"trung vị {fd(dc.so_ngay_tu_van_den_chot.median(), 0)} ngày.",
                                  f"Chỉ {fi(len(dc))}/{fi(WON)} ca chốt ({fp(len(dc) / WON * 100 if WON else 0)}) có ngày hợp lệ: kết quả chỉ mang tính tham khảo."],
                            pd.DataFrame({"Nhóm ngày": slabs, "Số ca": cnt.values, "% ca": np.round(cnt.values / cnt.sum() * 100, 1)}))

# ============================================================================
# 6. Du bao ML (toan bo du lieu)
# ============================================================================
ML_RES: dict = {}


def render_ml():
    ml = train_model(raw, (len(raw), int(df_all.chot.sum())))
    st.markdown(f'<div class="scope">Tab này luôn dùng toàn bộ dữ liệu (mô hình học trên {fi(ml["sizes"][0])} khách, '
                f'kiểm tra trên {fi(ml["sizes"][1])} khách, chấm điểm cuối trên {fi(ml["sizes"][2])} khách chưa từng thấy).</div>',
                unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Bài toán", "Cách kiểm chứng", "Mô hình", "Kết quả",
                                          "Yếu tố quan trọng", "Tương quan", "Thử dự báo"])
    with t1:
        c = st.columns(3)
        with c[0]: card("1. Dữ liệu đầu vào", [f"{fi(TOTAL_ALL)} khách đã biết kết quả", f"{fi(WON_ALL)} chốt, {fi(TOTAL_ALL - WON_ALL)} không chốt"])
        with c[1]: card("2. Huấn luyện mô hình", ["Tìm quy luật liên quan đến việc chốt", "Khóa học, tư vấn viên, thời điểm, chăm sóc"])
        with c[2]: card("3. Dự báo", ["Ước tính khả năng chốt của từng khách mới", "Ứng dụng: ưu tiên chăm sóc nhóm có khả năng chốt cao"])
        st.markdown("#### 12 thông tin đầu vào của mô hình")
        c = st.columns(3)
        with c[0]: card("Về khách", ["Khóa quan tâm", "Nhà mạng", "Cảm xúc (sentiment)", "Có thiếu tên hay không"])
        with c[1]: card("Về chăm sóc", ["Tư vấn viên phụ trách", "Mức chăm sóc", "Đã chuyển sang tư vấn chưa", "Số ngày đến khi tư vấn"])
        with c[2]: card("Về thời điểm", ["Tháng đăng ký", "Giờ đăng ký", "Thứ đăng ký", "Cuối tuần hay không"])
        st.caption("Không dùng ngày chốt sale và mã khách để tránh lộ đáp án cho mô hình.")
    with t2:
        a, b, c = ml["sizes"]
        k = st.columns(3)
        with k[0]: kpi("Train: huấn luyện mô hình", f"{fi(a)} khách")
        with k[1]: kpi("Test: kiểm tra khi xây dựng", f"{fi(b)} khách")
        with k[2]: kpi("Holdout: giữ riêng, chưa từng thấy", f"{fi(c)} khách")
        st.markdown("")
        tot_ = a + b + c
        note([f"Tỉ lệ chia thực tế: {fp(a / tot_ * 100)} / {fp(b / tot_ * 100)} / {fp(c / tot_ * 100)} (Train / Test / Holdout).",
              f"Chia có phân tầng nên tỉ lệ chốt gần bằng nhau ở cả ba phần: {fp(ml['pos'][0] / a * 100)}, {fp(ml['pos'][1] / b * 100)} và {fp(ml['pos'][2] / c * 100)}.",
              "Holdout không dùng để chọn biến hay chỉnh mô hình; chỉ dùng để chấm điểm cuối cùng."], "CÁCH LÀM")
    with t3:
        st.markdown("#### Random Forest: 300 cây quyết định, dự báo theo đa số phiếu")
        c = st.columns(4)
        with c[0]: card("Cây 1", ["Tư vấn viên nào?", "Đăng ký tháng mấy?", "<b>Phiếu: Chốt</b>"])
        with c[1]: card("Cây 2", ["Khóa nào?", "Mức chăm sóc nào?", "<b>Phiếu: Không chốt</b>"])
        with c[2]: card("Cây 3", ["Giờ đăng ký?", "Tư vấn viên nào?", "<b>Phiếu: Chốt</b>"])
        with c[3]: card("... đến cây 300", ["Mỗi cây nhìn dữ liệu khác nhau", "Kết quả theo đa số phiếu"])
        st.caption("Ví dụ minh họa cách bỏ phiếu. Kết hợp nhiều cây cho kết quả ổn định hơn một cây đơn lẻ.")
    with t4:
        thr = st.slider("Ngưỡng báo 'sẽ chốt' (xác suất từ mức này trở lên)", min_value=0.1, max_value=0.9, value=0.5, step=0.05, key="ml_thr")
        res = {k_: metrics_at(*v, thr) for k_, v in ml["proba"].items()}
        base = metrics_at(*ml["proba"]["Holdout 5%"], 0.5)
        ho, te = res["Holdout 5%"], res["Test 15%"]
        ML_RES["M"] = res
        ML_RES["thr"] = thr
        st.markdown("#### Kết quả trên Holdout 5% (dữ liệu mô hình chưa từng thấy)")
        k = st.columns(4)
        with k[0]: kpi("Accuracy: tỉ lệ dự báo đúng", fp(ho["Accuracy"] * 100))
        with k[1]: kpi("Recall: bắt được khách chốt thật", fp(ho["Recall"] * 100))
        with k[2]: kpi("Precision: đúng khi báo sẽ chốt", fp(ho["Precision"] * 100))
        with k[3]: kpi("ROC-AUC (không phụ thuộc ngưỡng)", fd(ho["ROC-AUC"], 3))
        st.markdown("")
        a, b = st.columns(2)
        with a:
            show(fig_cm(te["cm"], "Tập Test 15%"))
        with b:
            show(fig_cm(ho["cm"], "Tập Holdout 5%"))
        tn, fp_, fn, tp = ho["cm"].ravel()
        n_ho = int(ho["cm"].sum())
        bl = [f"Holdout ở ngưỡng {fp(thr * 100, 0)}: dự báo đúng {fi(tn + tp)}/{fi(n_ho)} khách ({fp(ho['Accuracy'] * 100)}).",
              f"Bắt được {fi(tp)}/{fi(tp + fn)} khách chốt thật; báo nhầm {fi(fp_)} khách; bỏ sót {fi(fn)} khách."]
        if abs(thr - 0.5) > 1e-9:
            bl.append(f"So với ngưỡng 50%: Recall {fp(base['Recall'] * 100)} → {fp(ho['Recall'] * 100)}, "
                      f"Precision {fp(base['Precision'] * 100)} → {fp(ho['Precision'] * 100)}. "
                      + ("Hạ ngưỡng bắt thêm khách chốt nhưng báo nhầm nhiều hơn." if thr < 0.5
                         else "Nâng ngưỡng giảm báo nhầm nhưng bỏ sót nhiều khách chốt hơn."))
        else:
            bl.append("Ở ngưỡng 50%, mô hình ưu tiên không bỏ sót khách chốt." if ho["Recall"] > ho["Precision"]
                      else "Ở ngưỡng 50%, mô hình ưu tiên độ chính xác khi báo sẽ chốt.")
        dacc = abs(te["Accuracy"] - ho["Accuracy"]) * 100
        bl.append(f"Test {fp(te['Accuracy'] * 100)} và Holdout {fp(ho['Accuracy'] * 100)}: " +
                  ("gần bằng nhau, kết quả ổn định." if dacc < 2 else f"chênh {fd(dacc, 1)} điểm phần trăm."))
        note(bl)
        st.markdown("#### So sánh ba tập")
        tbl = pd.DataFrame({k_: {m_: (round(v[m_] * 100, 1) if m_ != "ROC-AUC" else round(v[m_], 3))
                                 for m_ in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]} for k_, v in res.items()})
        st.dataframe(tbl)
        st.markdown("#### Đối chiếu từng khách trong Holdout")
        y_ho, p_ho = ml["proba"]["Holdout 5%"]
        hold = ml["meta"].copy()
        hold["Thực tế"] = np.where(y_ho == 1, "Chốt", "Không chốt")
        hold["Dự báo"] = np.where(p_ho >= thr, "Chốt", "Không chốt")
        hold["Xác suất chốt"] = np.round(p_ho, 3)
        hold["Kết quả"] = np.where(hold["Thực tế"] == hold["Dự báo"], "Đúng", "Sai")
        c1, c2 = st.columns(2)
        with c1:
            flt = st.radio("Hiển thị", ["Tất cả", "Chỉ dự báo sai", "Chỉ dự báo đúng"], horizontal=True, key="ho_flt")
        with c2:
            prange = st.slider("Khoảng xác suất chốt", min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.05, key="ho_prob")
        hd = hold[hold["Xác suất chốt"].between(prange[0], prange[1])]
        if flt == "Chỉ dự báo sai":
            hd = hd[hd["Kết quả"] == "Sai"]
        elif flt == "Chỉ dự báo đúng":
            hd = hd[hd["Kết quả"] == "Đúng"]
        st.caption(f"Đang hiển thị {fi(len(hd))}/{fi(len(hold))} khách; trong đó dự báo đúng {fi((hd['Kết quả'] == 'Đúng').sum())}.")
        st.dataframe(hd.sort_values("Xác suất chốt", ascending=False), hide_index=True, height=360)
        st.download_button("Tải bảng đối chiếu (CSV)", data=hold.to_csv(index=False).encode("utf-8-sig"),
                           file_name="holdout_du_bao_vs_thuc_te.csv", mime="text/csv")
    with t5:
        imp = ml["imp"]
        topi = slider_n("Số yếu tố hiển thị (Top)", 3, len(imp), min(8, len(imp)), "imp_n")
        d = imp.head(topi)
        fig, _ = hbar(list(d.index), d.values * 100, [TEAL if i < 2 else GREY for i in range(len(d))], lambda i, v: fp(v),
                      "Mức quan trọng (%)", f"Top {len(d)} yếu tố ảnh hưởng nhiều nhất đến dự báo", xmax=max(d.values * 100) * 1.25)
        chart_with_note(fig, [f"{imp.index[0]} ({fp(imp.iloc[0] * 100)}) và {imp.index[1]} ({fp(imp.iloc[1] * 100)}): gộp {fp((imp.iloc[0] + imp.iloc[1]) * 100, 0)} tổng mức quan trọng.",
                              f"Top {len(d)} yếu tố chiếm {fp(d.sum() * 100, 0)} tổng mức quan trọng.",
                              f"Ít ảnh hưởng nhất: {imp.index[-1]} ({fp(imp.iloc[-1] * 100)}).",
                              "Mô hình dựa vào các yếu tố này để dự báo, chưa chắc là nguyên nhân dẫn đến việc chốt."],
                        pd.DataFrame({"Yếu tố": d.index, "Mức quan trọng (%)": np.round(d.values * 100, 2)}))
    with t6:
        corr = ml["corr"]
        view = st.radio("Hiển thị", ["Ma trận đầy đủ", "Chỉ tương quan với việc chốt"], horizontal=True, key="corr_view")
        ch = corr["chot"].drop("chot")
        top_ = ch.abs().sort_values(ascending=False)
        if view == "Ma trận đầy đủ":
            c1, c2 = st.columns([1.5, 1])
            with c1:
                show(fig_corr(corr))
            with c2:
                sub = corr.drop(index="chot", columns="chot")
                best_pair, best_val = None, -1.0
                cols_ = list(sub.columns)
                for i in range(len(cols_)):
                    for j in range(i + 1, len(cols_)):
                        v_ = abs(float(sub.iloc[i, j]))
                        if v_ > best_val:
                            best_pair, best_val = (cols_[i], cols_[j]), v_
                note([f"Tương quan càng gần 0 càng ít liên quan. Với việc chốt, cao nhất là {VAR_NAMES.get(top_.index[0], top_.index[0]).lower()} "
                      f"({fd(ch[top_.index[0]])}), các biến còn lại dưới {fd(float(np.ceil(top_.iloc[1] * 100) / 100))}.",
                      f"Cặp biến liên hệ chặt nhất: {VAR_NAMES.get(best_pair[0], best_pair[0]).lower()} và "
                      f"{VAR_NAMES.get(best_pair[1], best_pair[1]).lower()} ({fd(best_val)}).",
                      "Không thông tin đơn lẻ nào quyết định việc chốt: cần mô hình kết hợp nhiều thông tin."])
        else:
            d = ch.reindex(top_.index)
            fig, ax = new_fig(8, 4.6)
            ax.barh(range(len(d)), d.values, color=[TEAL if v > 0 else ORANGE for v in d.values], height=0.6)
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels([VAR_NAMES.get(i, i) for i in d.index], fontsize=11)
            ax.invert_yaxis()
            ax.axvline(0, color=INK, lw=1)
            for i, v in enumerate(d.values):
                ax.text(v + (0.004 if v >= 0 else -0.004), i, fd(v), va="center", ha="left" if v >= 0 else "right",
                        fontsize=10, fontweight="bold")
            ax.set_xlabel("Hệ số tương quan với việc chốt", fontsize=11, fontweight="bold", color="#5D656C")
            ax.set_title("Tương quan của từng biến với việc chốt", fontsize=14, fontweight="bold", loc="left", pad=12)
            style(ax, "x")
            fig.tight_layout()
            chart_with_note(fig, [f"Cao nhất: {VAR_NAMES.get(top_.index[0], top_.index[0]).lower()} ({fd(ch[top_.index[0]])}).",
                                  f"{int((top_ < 0.05).sum())}/{len(top_)} biến có |tương quan| dưới 0,05: gần như không liên quan tuyến tính.",
                                  "Tương quan chỉ đo quan hệ tuyến tính; mô hình cây vẫn khai thác được quan hệ phức tạp hơn."],
                            pd.DataFrame({"Biến": [VAR_NAMES.get(i, i) for i in d.index], "Tương quan": np.round(d.values, 3)}))
    with t7:
        st.markdown("#### Thử dự báo cho một khách giả định")
        st.caption("Chỉ mang tính minh họa. Mô hình dựa nhiều vào tư vấn viên và tháng đăng ký.")
        gtt = tv_table(df_all)
        tvv_opts = list(gtt[gtt.n >= 30].sort_values("n", ascending=False).index)
        course_opts = list(raw["quan_tam_khoa_hoc"].dropna().value_counts().index)
        net_opts = list(raw["nha_mang"].dropna().value_counts().index)
        care_opts = list(raw["MKT_xu_ly_data"].dropna().value_counts().index)
        a, b, c = st.columns(3)
        with a:
            khoa = st.selectbox("Khóa quan tâm", course_opts, index=0, key="p_khoa")
            tvv = st.selectbox("Tư vấn viên", tvv_opts, index=0, key="p_tvv")
            net = st.selectbox("Nhà mạng", net_opts, index=0, key="p_net")
        with b:
            sent = st.selectbox("Sentiment", list(SENT_NAMES), index=2, format_func=lambda x: SENT_NAMES[x], key="p_sent")
            care = st.selectbox("Mức chăm sóc", care_opts, index=0, key="p_care")
            month = st.selectbox("Tháng đăng ký", list(range(1, 13)), index=5, key="p_month")
        with c:
            hour = st.selectbox("Giờ đăng ký", list(range(0, 24)), index=9, key="p_hour")
            wd = st.selectbox("Thứ đăng ký", list(range(7)), index=0, format_func=lambda x: WEEKDAYS[x], key="p_wd")
            has_c = st.checkbox("Đã chuyển sang tư vấn", value=False, key="p_has")
            days_c = st.number_input("Số ngày đến khi tư vấn", min_value=0.0, max_value=200.0, value=1.0, key="p_days") if has_c else np.nan
        row = pd.DataFrame([{"MKT_xu_ly_data": care, "nha_mang": net, "quan_tam_khoa_hoc": khoa, "sentimatic": sent,
                             "tu_van_vien": tvv, "loi_ten_rong": 0, "has_consult": int(has_c), "days_to_consult": days_c,
                             "signup_hour": hour, "signup_weekday": wd, "signup_month": month,
                             "signup_weekend": int(wd >= 5)}])[CAT_COLS + NUM_COLS]
        p = float(ml["model"].predict_proba(ml["pre"].transform(row))[:, 1][0])
        thr_ = ML_RES.get("thr", 0.5)
        k = st.columns(3)
        with k[0]: kpi("Xác suất chốt ước tính", fp(p * 100))
        with k[1]: kpi("Trung bình toàn bộ dữ liệu", fp(TB_ALL))
        with k[2]: kpi(f"Dự báo (ngưỡng {fp(thr_ * 100, 0)})", "Chốt" if p >= thr_ else "Không chốt")
        st.progress(min(max(p, 0.0), 1.0))


with tabs[5]:
    st.markdown("### Dự báo khách chốt sale bằng Machine Learning")
    if st.session_state.get("ml_ready", False):
        render_ml()
    else:
        st.info("Mô hình Random Forest được huấn luyện trực tiếp trên dữ liệu (khoảng 20-40 giây). "
                "Bấm nút bên dưới để chạy; các tab khác không bị chậm.")
        if st.button("Chạy mô hình dự báo", key="run_ml"):
            st.session_state["ml_ready"] = True
            st.rerun()

# ============================================================================
# 7. Ket luan (tinh tren toan bo du lieu)
# ============================================================================
with tabs[6]:
    st.markdown("### Kết luận và đề xuất")
    st.markdown(f'<div class="scope">Kết luận dựa trên toàn bộ {fi(TOTAL_ALL)} khách (không áp dụng bộ lọc).</div>', unsafe_allow_html=True)
    gca = course_table(df_all)
    gca = gca.assign(score=0.5 * minmax(gca.n) + 0.5 * minmax(gca.rate)).sort_values("score", ascending=False)
    gta = tv_table(df_all)
    solid_all = gta[gta.n >= SOLID_N].sort_values("rate", ascending=False)
    won_nc = int(((df_all.chot == 1) & (df_all.co_tu_van == 0)).sum())
    sg = df_all.groupby("sent").chot.mean() * 100
    find = [f"{fp(won_nc / WON_ALL * 100, 0)} khách chốt không qua bước tư vấn" +
            (": có ít nhất hai luồng bán hàng." if won_nc / WON_ALL >= 0.5 else ".")]
    best_c = gca.index[0]
    find.append(f"{best_c} là khóa có điểm tiềm năng cao nhất ({fd(gca.score.iloc[0], 3)}): {fi(gca.n[best_c])} khách, chốt {fp(gca.rate[best_c])}.")
    weak = gca[gca.n >= 500].sort_values("rate")
    if len(weak):
        find.append(f"{weak.index[0]} có {fi(weak.n.iloc[0])} khách nhưng chỉ chốt {fp(weak.rate.iloc[0])}.")
    if len(solid_all) >= 2:
        find.append(f"Trong tư vấn viên có từ {SOLID_N} khách: cao nhất {fp(solid_all.rate.iloc[0])} ({solid_all.index[0]}), "
                    f"thấp nhất {fp(solid_all.rate.iloc[-1])} ({solid_all.index[-1]}).")
    if {"Tích cực", "Tiêu cực"} <= set(sg.index):
        dd = abs(sg["Tích cực"] - sg["Tiêu cực"])
        find.append(f"Sentiment tích cực {fp(sg['Tích cực'])} so với tiêu cực {fp(sg['Tiêu cực'])}: " +
                    ("gần bằng nhau, không phân biệt được khách chốt." if dd < 2 else f"chênh {fd(dd, 1)} điểm phần trăm."))
    Mx = ML_RES.get("M")
    if Mx is not None:
        ho_ = Mx["Holdout 5%"]
        mod = [f"Đúng {fp(ho_['Accuracy'] * 100)} trên dữ liệu chưa từng thấy (ngưỡng {fp(ML_RES.get('thr', 0.5) * 100, 0)}).",
               f"Bắt {fp(ho_['Recall'] * 100, 0)} khách chốt thật; Precision {fp(ho_['Precision'] * 100, 0)}."]
    else:
        mod = ["Chạy mô hình ở tab 6 để xem kết quả."]
    mod.append("Phụ thuộc mạnh vào tư vấn viên và tháng đăng ký.")
    c = st.columns(3)
    with c[0]: card("Phát hiện", find)
    with c[1]: card("Mô hình", mod)
    with c[2]:
        card("Đề xuất", ["Rà soát các tư vấn viên có tỉ lệ chốt rất thấp.", "Chia lead theo hiệu suất từng người.",
                         "Bổ sung dữ liệu kênh marketing để đánh giá nguồn khách.",
                         "Kiểm tra mô hình theo thời gian (train tháng cũ, kiểm tra tháng mới)."])
    st.markdown("#### Hạn chế cần lưu ý")
    valid_all = int(((df_all.chot == 1) & df_all.so_ngay_tu_van_den_chot.notna()).sum())
    note([f"Dữ liệu trải trên {len(months_all)} tháng; cần ít nhất hai năm mới kết luận chắc về tính mùa vụ.",
          "Chưa có dữ liệu kênh marketing; cột nhà mạng chỉ là nhà mạng SIM.",
          f"Chỉ {fp(valid_all / WON_ALL * 100, 0)} ca chốt có ngày hợp lệ nên phân tích tốc độ chốt chỉ mang tính tham khảo.",
          "Mô hình cho biết yếu tố liên quan, không chứng minh nguyên nhân."], "HẠN CHẾ")