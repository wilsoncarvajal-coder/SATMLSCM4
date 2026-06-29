"""
==============================================================
app.py — Dashboard Ejecutivo SAT-ML SCM v2.0
--------------------------------------------------------------
Sistema Inteligente de Analítica Predictiva Logística
basado en el Modelo SAT-ML para la Optimización de Inventarios
en Centros de Distribución (CEDIs).

Trabajo de Grado 3 — Modelos de Innovación
Ingeniería de Sistemas — CUN: Corporación Unificada Nacional de Educación Superior

Autor : Wilson Andrés Carvajal Barreto
Versión: 2.0.0
==============================================================
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────
# STDLIB
# ──────────────────────────────────────────────────────────
import io
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────
# THIRD-PARTY
# ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

# ──────────────────────────────────────────────────────────
# LOCAL MODULES
# ──────────────────────────────────────────────────────────
from config import (
    APP_NAME, APP_VERSION, AUTHOR, UNIVERSIDAD,
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    RANDOM_STATE, TEST_SIZE,
    TARGET_COLUMN, RF_PARAMS,
    CEDIS, DIAS_HISTORICOS,
    PRIMARY_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR,
    DEFAULT_USERS, FOOTER, HORAS_PROYECCION, EXPORT_DIR,
)
from generator import DataGenerator
from utils.logger import LoggerManager

# ──────────────────────────────────────────────────────────
# LOGGER
# ──────────────────────────────────────────────────────────
logger = LoggerManager.get_logger(__name__)

# ──────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# ESTILOS CSS PERSONALIZADOS
# ══════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ─── Fuente base ─── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0F2044 100%);
    }
    [data-testid="stSidebar"] * { color: #e8edf5 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label { color: #94a3c0 !important; }

    /* ─── KPI Cards ─── */
    .kpi-card {
        background: linear-gradient(135deg, #1a2740 0%, #0f1e38 100%);
        border: 1px solid #2a3f66;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
        transition: transform .2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-value   { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .kpi-label   { font-size: .82rem; color: #94a3c0; margin-top: 4px; }
    .kpi-delta   { font-size: .78rem; margin-top: 6px; }
    .kpi-up   { color: #34d399; }
    .kpi-down { color: #f87171; }

    /* ─── Section headers ─── */
    .section-header {
        border-left: 4px solid #3b82f6;
        padding-left: 12px;
        margin: 24px 0 14px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* ─── Alert badges ─── */
    .badge-alto   { background:#7f1d1d; color:#fca5a5; border-radius:6px; padding:2px 10px; }
    .badge-medio  { background:#78350f; color:#fcd34d; border-radius:6px; padding:2px 10px; }
    .badge-bajo   { background:#14532d; color:#86efac; border-radius:6px; padding:2px 10px; }

    /* ─── Login card ─── */
    .login-card {
        max-width: 420px;
        margin: 60px auto;
        background: #1a2740;
        border-radius: 16px;
        padding: 40px 36px;
        box-shadow: 0 8px 40px rgba(0,0,0,.5);
        border: 1px solid #2a3f66;
    }

    /* ─── Divider ─── */
    .hr { border: none; border-top: 1px solid #2a3f66; margin: 18px 0; }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        font-size: .75rem;
        color: #64748b;
        padding: 24px 0 8px;
        border-top: 1px solid #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════
# UTILIDADES GENERALES
# ══════════════════════════════════════════════════════════

def _badge(riesgo: str) -> str:
    cls = {"Alto": "badge-alto", "Medio": "badge-medio", "Bajo": "badge-bajo"}.get(riesgo, "")
    return f'<span class="{cls}">{riesgo}</span>'


def _kpi(label: str, value: str, delta: str = "", up: bool = True) -> str:
    direction = "kpi-up" if up else "kpi-down"
    delta_html = f'<div class="kpi-delta {direction}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
      {delta_html}
    </div>"""


def _section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# AUTENTICACIÓN
# ══════════════════════════════════════════════════════════

def login() -> bool:
    """Muestra el formulario de inicio de sesión."""

    st.markdown(
        """
        <div class="login-card">
          <h2 style="color:#60a5fa;text-align:center;margin-bottom:8px;">📦 SAT-ML SCM</h2>
          <p style="color:#94a3c0;text-align:center;font-size:.88rem;margin-bottom:24px;">
            Sistema Inteligente de Analítica Predictiva Logística
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        usuario = st.text_input("👤 Usuario", placeholder="admin / analista / coordinador")
        clave   = st.text_input("🔑 Contraseña", type="password")
        submit  = st.form_submit_button("Ingresar al sistema", use_container_width=True)

    if submit:
        if usuario in DEFAULT_USERS and DEFAULT_USERS[usuario]["password"] == clave:
            st.session_state["authenticated"] = True
            st.session_state["usuario"]  = usuario
            st.session_state["rol"]      = DEFAULT_USERS[usuario]["role"]
            logger.info("Usuario '%s' autenticado.", usuario)
            st.rerun()
        else:
            st.error("Credenciales incorrectas. Intenta de nuevo.")
    return False


# ══════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════

@st.cache_data(show_spinner="⏳ Cargando datos históricos…")
def cargar_datos(uploaded_file=None) -> pd.DataFrame:
    """
    Carga el histórico desde:
      1. Archivo subido por el usuario.
      2. CSV en data/historico_cedis.csv (incluido en el repo).
      3. Generador sintético de DataGenerator.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["Fecha"])
        logger.info("CSV subido por usuario cargado (%s registros).", len(df))
        return df

    csv_path = Path("data/historico_cedis.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["Fecha"])
        logger.info("CSV local cargado (%s registros).", len(df))
        return df

    gen = DataGenerator(dias=DIAS_HISTORICOS)
    df  = gen.generar()
    logger.info("Datos sintéticos generados (%s registros).", len(df))
    return df


# ══════════════════════════════════════════════════════════
# PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════

def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


# ══════════════════════════════════════════════════════════
# ENTRENAMIENTO Y COMPARACIÓN
# ══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🤖 Entrenando modelos…")
def entrenar_modelos(df_hash: int, _df: pd.DataFrame):
    """
    Entrena 4 modelos, compara métricas y devuelve el mejor.
    Se usa df_hash para invalidar la caché cuando cambien los datos.
    """
    # Preparar features
    df_ml = _df.copy()

    # Eliminar columnas no útiles para el modelo
    drop_cols = [c for c in ["ID_Despacho", "Fecha", TARGET_COLUMN, "Quiebre_Stock"] if c in df_ml.columns]
    X = df_ml.drop(columns=drop_cols)
    y = df_ml[TARGET_COLUMN] if TARGET_COLUMN in df_ml.columns else df_ml["Demanda_Real"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = _build_preprocessor(X_train)

    candidatos = {
        "Regresión Lineal":    LinearRegression(),
        "Árbol de Decisión":   DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=8),
        "Gradient Boosting":   GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=200),
        "Random Forest ⭐":    RandomForestRegressor(**RF_PARAMS),
    }

    resultados = []
    pipelines  = {}

    for nombre, modelo in candidatos.items():
        t0 = time.perf_counter()
        pipe = Pipeline([("pre", preprocessor), ("model", modelo)])
        pipe.fit(X_train, y_train)
        t1 = time.perf_counter()

        y_pred = pipe.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2   = r2_score(y_test, y_pred)
        acc  = max(0, 100 - mape)

        resultados.append({
            "Modelo":       nombre,
            "MAE":          round(mae, 2),
            "RMSE":         round(rmse, 2),
            "MAPE (%)":     round(mape, 2),
            "R²":           round(r2, 4),
            "Precisión (%)":round(acc, 2),
            "Tiempo (s)":   round(t1 - t0, 3),
        })
        pipelines[nombre] = (pipe, y_test, y_pred)

    comp_df = pd.DataFrame(resultados).sort_values("R²", ascending=False).reset_index(drop=True)
    best_name = comp_df.iloc[0]["Modelo"]
    best_pipe, best_y_test, best_y_pred = pipelines[best_name]

    logger.info("Mejor modelo: %s  R²=%.4f", best_name, comp_df.iloc[0]["R²"])
    return best_pipe, comp_df, best_name, best_y_test, best_y_pred, X_test


# ══════════════════════════════════════════════════════════
# PROYECCIÓN DE DEMANDA (72 h)
# ══════════════════════════════════════════════════════════

def _proyectar(df: pd.DataFrame, modelo, cedi_sel: str) -> pd.DataFrame:
    """Genera tabla de proyección para las próximas 72 horas (3 días)."""
    df_cedi = df[df["CEDI"] == cedi_sel].copy().sort_values("Fecha")
    last = df_cedi.iloc[-1]

    registros = []
    for h in range(0, HORAS_PROYECCION, 24):
        fila = last.copy()
        if "Dia_Semana" in fila:
            fila["Dia_Semana"] = (int(fila["Dia_Semana"]) + h // 24) % 7
        if "Tiempo_Trafico_Min" in fila:
            fila["Tiempo_Trafico_Min"] = float(fila["Tiempo_Trafico_Min"]) * np.random.uniform(0.9, 1.15)

        drop = [c for c in ["ID_Despacho", "Fecha", TARGET_COLUMN, "Demanda_Real", "Quiebre_Stock"] if c in fila.index]
        X_row = pd.DataFrame([fila.drop(labels=drop)])
        demanda_pred = float(modelo.predict(X_row)[0])

        stock_col = "Stock_Disponible" if "Stock_Disponible" in last.index else "Stock_Inicial"
        stock_val  = float(last[stock_col]) if stock_col in last.index else 300.0
        stock_rec  = demanda_pred * 1.15

        relacion = stock_val / demanda_pred if demanda_pred > 0 else 1.5
        if relacion >= 1.20:
            riesgo = "Bajo"
        elif relacion >= 1.00:
            riesgo = "Medio"
        else:
            riesgo = "Alto"

        registros.append({
            "Horizonte":          f"T+{h+24}h",
            "CEDI":               cedi_sel,
            "Demanda Predicha":   round(demanda_pred),
            "Stock Disponible":   round(stock_val),
            "Stock Recomendado":  round(stock_rec),
            "Diferencia":         round(stock_val - stock_rec),
            "Riesgo":             riesgo,
            "Alerta":             {
                "Alto":  "⛔ Reposición inmediata",
                "Medio": "⚠️ Monitorear inventario",
                "Bajo":  "✅ Inventario suficiente",
            }[riesgo],
        })

    return pd.DataFrame(registros)


# ══════════════════════════════════════════════════════════
# EXPORTACIÓN EXCEL
# ══════════════════════════════════════════════════════════

def exportar_excel(df_hist: pd.DataFrame, df_comp: pd.DataFrame, df_proy: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_hist.to_excel(writer, sheet_name="Histórico",     index=False)
        df_comp.to_excel(writer, sheet_name="Comparación",   index=False)
        df_proy.to_excel(writer, sheet_name="Proyección 72h", index=False)

        wb = writer.book
        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#0F2044", "font_color": "white",
            "border": 1, "align": "center",
        })
        for sheet_name, df_ in [
            ("Histórico", df_hist),
            ("Comparación", df_comp),
            ("Proyección 72h", df_proy),
        ]:
            ws = writer.sheets[sheet_name]
            for col_num, col_name in enumerate(df_.columns):
                ws.write(0, col_num, col_name, header_fmt)
                ws.set_column(col_num, col_num, max(14, len(str(col_name)) + 4))

    logger.info("Excel exportado.")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════
# EXPORTACIÓN PDF
# ══════════════════════════════════════════════════════════

def exportar_pdf(df_comp: pd.DataFrame, df_proy: pd.DataFrame, model_name: str) -> bytes:
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # Título
    story.append(Paragraph(
        f"<font color='#0F62FE'><b>SAT-ML SCM v{APP_VERSION}</b></font> — Informe Ejecutivo",
        styles["Title"]
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Autor: {AUTHOR} | {UNIVERSIDAD} | Mejor modelo: {model_name}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.5*cm))

    # Tabla comparación
    story.append(Paragraph("<b>Comparación de Modelos</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2*cm))
    comp_data = [df_comp.columns.tolist()] + df_comp.values.tolist()
    t_comp = Table(comp_data, repeatRows=1)
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0F2044")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#f0f4ff"), rl_colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#CBD5E0")),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 0.6*cm))

    # Tabla proyección
    story.append(Paragraph("<b>Proyección de Demanda — Próximas 72 horas</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2*cm))
    proy_data = [df_proy.columns.tolist()] + df_proy.values.tolist()
    t_proy = Table(proy_data, repeatRows=1)
    t_proy.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0F2044")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#f0f4ff"), rl_colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#CBD5E0")),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t_proy)

    doc.build(story)
    logger.info("PDF exportado.")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center;padding:16px 0 8px;">
              <span style="font-size:2.4rem;">📦</span><br>
              <span style="font-size:1.1rem;font-weight:700;color:#60a5fa;">SAT-ML SCM</span><br>
              <span style="font-size:.72rem;color:#64748b;">v{APP_VERSION}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Usuario
        st.markdown(
            f"""
            <div style="background:#0a1e3d;border-radius:8px;padding:10px 14px;margin-bottom:12px;">
              <span style="font-size:.8rem;color:#94a3c0;">Usuario</span><br>
              <b>{st.session_state.get("usuario","—")}</b><br>
              <span style="font-size:.75rem;color:#3b82f6;">{st.session_state.get("rol","—")}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Carga de datos ──
        st.markdown("#### 📂 Datos")
        uploaded = st.file_uploader(
            "Subir CSV propio",
            type=["csv"],
            help="Formato: Fecha, CEDI, Demanda_Real, Stock_Disponible, …",
        )

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # ── Filtros ──
        st.markdown("#### 🔍 Filtros")

        cedis_disp = ["Todos"] + sorted(df["CEDI"].unique().tolist())
        cedi_sel   = st.selectbox("📍 CEDI", cedis_disp)

        cats_disp = ["Todas"]
        cat_col   = next((c for c in ["Categoria_Producto", "Categoria"] if c in df.columns), None)
        if cat_col:
            cats_disp += sorted(df[cat_col].dropna().unique().tolist())
        cat_sel = st.selectbox("🏷️ Categoría", cats_disp)

        if "Fecha" in df.columns:
            df["Fecha"] = pd.to_datetime(df["Fecha"])
            min_f = df["Fecha"].min().date()
            max_f = df["Fecha"].max().date()
            rango = st.date_input("📅 Rango de fechas", (min_f, max_f))
        else:
            rango = None

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # ── Navegación ──
        st.markdown("#### 🧭 Secciones")
        seccion = st.radio(
            "",
            ["🏠 Dashboard Ejecutivo", "🤖 Entrenamiento", "📈 Predicción 72h",
             "🚨 Alertas", "📊 Análisis Exploratorio", "ℹ️ Acerca de"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Logout
        if st.button("🚪 Cerrar sesión", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    return {
        "uploaded": uploaded,
        "cedi":     cedi_sel,
        "cat":      cat_sel,
        "rango":    rango,
        "seccion":  seccion,
        "cat_col":  cat_col,
    }


# ══════════════════════════════════════════════════════════
# SECCIÓN: DASHBOARD EJECUTIVO
# ══════════════════════════════════════════════════════════

def sec_dashboard(df: pd.DataFrame, opts: dict) -> None:
    st.markdown(
        f"<h2 style='color:#60a5fa;'>🏠 Dashboard Ejecutivo — {APP_NAME}</h2>",
        unsafe_allow_html=True,
    )

    df_f = _filtrar(df, opts)

    # ── KPIs ──
    total_reg    = len(df_f)
    col_demanda  = "Demanda_Real" if "Demanda_Real" in df_f.columns else df_f.select_dtypes("number").columns[0]
    col_stock    = next((c for c in ["Stock_Disponible","Stock_Inicial"] if c in df_f.columns), None)
    col_sat      = next((c for c in ["Satisfaccion_SAT","Indice_Satisfaccion_SAT"] if c in df_f.columns), None)
    col_desp     = next((c for c in ["Desperdicio_Kg"] if c in df_f.columns), None)

    avg_demanda  = df_f[col_demanda].mean() if col_demanda in df_f.columns else 0
    avg_stock    = df_f[col_stock].mean()   if col_stock   else 0
    avg_sat      = df_f[col_sat].mean()     if col_sat     else 0
    perc_quiebre = (
        (df_f["Quiebre_Stock"] == "Si").mean() * 100
        if "Quiebre_Stock" in df_f.columns else 0
    )
    desp_total   = df_f[col_desp].sum() if col_desp else 0

    cols = st.columns(5)
    kpis = [
        ("Registros analizados", f"{total_reg:,}", "", True),
        ("Demanda promedio",     f"{avg_demanda:,.0f}", "unidades/día", True),
        ("Stock promedio",       f"{avg_stock:,.0f}",  "unidades",     True),
        ("SAT promedio",         f"{avg_sat:.2f}/5.0", "satisfacción", avg_sat >= 4.0),
        ("% Quiebre de stock",   f"{perc_quiebre:.1f}%", "⚠️ a reducir", False),
    ]
    for col, (lbl, val, delta, up) in zip(cols, kpis):
        col.markdown(_kpi(lbl, val, delta, up), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráficos principales ──
    col1, col2 = st.columns(2)

    with col1:
        _section("📈 Demanda histórica por CEDI")
        if "Fecha" in df_f.columns:
            df_line = df_f.groupby(["Fecha","CEDI"])[col_demanda].mean().reset_index()
            fig = px.line(df_line, x="Fecha", y=col_demanda, color="CEDI",
                          color_discrete_sequence=["#3b82f6","#f59e0b"],
                          template="plotly_dark", markers=False)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              legend_title_text="CEDI", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        _section("📊 Distribución de demanda")
        fig2 = px.box(df_f, x="CEDI", y=col_demanda, color="CEDI",
                      color_discrete_sequence=["#3b82f6","#f59e0b"],
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if col_sat and "CEDI" in df_f.columns:
            _section("😊 SAT vs Demanda")
            fig3 = px.scatter(df_f, x=col_demanda, y=col_sat, color="CEDI",
                              color_discrete_sequence=["#3b82f6","#f59e0b"],
                              template="plotly_dark", opacity=0.6,
                              trendline="ols")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if "Quiebre_Stock" in df_f.columns:
            _section("🚦 Quiebres de stock")
            df_qb = df_f.groupby(["CEDI","Quiebre_Stock"]).size().reset_index(name="N")
            fig4  = px.bar(df_qb, x="CEDI", y="N", color="Quiebre_Stock",
                           barmode="group", template="plotly_dark",
                           color_discrete_map={"Si":"#ef4444","No":"#22c55e"})
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Datos crudos ──
    with st.expander("🗄️ Ver tabla de datos filtrados"):
        st.dataframe(df_f.head(200), use_container_width=True)


# ══════════════════════════════════════════════════════════
# SECCIÓN: ENTRENAMIENTO
# ══════════════════════════════════════════════════════════

def sec_entrenamiento(df: pd.DataFrame) -> tuple:
    st.markdown("<h2 style='color:#60a5fa;'>🤖 Entrenamiento y Comparación de Modelos</h2>",
                unsafe_allow_html=True)

    col_target = "Demanda_Real" if "Demanda_Real" in df.columns else TARGET_COLUMN
    if col_target not in df.columns:
        st.error(f"No se encontró la columna objetivo '{col_target}' en el dataset.")
        return None, None, None

    st.info(
        f"Se entrenan **4 modelos** sobre **{len(df):,} registros** con partición {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} "
        f"train/test y semilla `{RANDOM_STATE}`. El mejor modelo (mayor R²) se usa para todas las predicciones."
    )

    df_hash  = hash(str(df.shape) + str(df.columns.tolist()))
    best_pipe, comp_df, best_name, y_test, y_pred, X_test = entrenar_modelos(df_hash, df)

    # ── Tabla comparación ──
    _section("📋 Tabla comparativa de modelos")

    def _color_best(row):
        styles = [""] * len(row)
        if row["Modelo"] == best_name:
            styles = ["background-color:#14532d;color:#86efac;font-weight:700"] * len(row)
        return styles

    st.dataframe(
        comp_df.style.apply(_color_best, axis=1).format({
            "MAE": "{:.2f}", "RMSE": "{:.2f}",
            "MAPE (%)": "{:.2f}", "R²": "{:.4f}",
            "Precisión (%)": "{:.2f}", "Tiempo (s)": "{:.3f}",
        }),
        use_container_width=True,
    )

    best_row = comp_df[comp_df["Modelo"] == best_name].iloc[0]
    st.success(
        f"✅ **Mejor modelo:** {best_name} — R² = {best_row['R²']:.4f} | "
        f"MAPE = {best_row['MAPE (%)']:.2f}% | Precisión = {best_row['Precisión (%)']:.2f}%"
    )

    # ── Gráficos ──
    col1, col2 = st.columns(2)

    with col1:
        _section("🎯 Predicho vs Real (mejor modelo)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(y_test))), y=y_test.values,
            mode="lines", name="Real", line=dict(color="#60a5fa", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))), y=y_pred,
            mode="lines", name="Predicho", line=dict(color="#f59e0b", width=2, dash="dash")
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=30,b=0),
                          legend_title_text="Serie")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        _section("📊 R² por modelo")
        fig2 = px.bar(comp_df, x="Modelo", y="R²", color="R²",
                      color_continuous_scale="Blues", template="plotly_dark",
                      text="R²")
        fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           showlegend=False, margin=dict(l=0,r=0,t=30,b=0),
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    return best_pipe, comp_df, best_name


# ══════════════════════════════════════════════════════════
# SECCIÓN: PREDICCIÓN 72h
# ══════════════════════════════════════════════════════════

def sec_prediccion(df: pd.DataFrame, modelo, best_name: str, opts: dict) -> pd.DataFrame:
    st.markdown("<h2 style='color:#60a5fa;'>📈 Predicción de Demanda — Próximas 72 horas</h2>",
                unsafe_allow_html=True)

    if modelo is None:
        st.warning("Primero entrena los modelos en la sección correspondiente.")
        return pd.DataFrame()

    cedi_proy = opts["cedi"] if opts["cedi"] != "Todos" else CEDIS[0]
    st.info(f"Proyección para **CEDI {cedi_proy}** usando **{best_name}**")

    df_proy = _proyectar(df, modelo, cedi_proy)

    # KPIs de proyección
    cols = st.columns(3)
    for i, row in df_proy.iterrows():
        badge = {"Alto":"🔴","Medio":"🟡","Bajo":"🟢"}[row["Riesgo"]]
        cols[i % 3].markdown(
            _kpi(
                f"{badge} {row['Horizonte']}",
                f"{row['Demanda Predicha']:,}",
                row["Alerta"],
                row["Riesgo"] != "Alto",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico de barras con umbrales
    _section("📊 Demanda predicha vs stock disponible")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_proy["Horizonte"], y=df_proy["Demanda Predicha"],
        name="Demanda Predicha", marker_color="#3b82f6"
    ))
    fig.add_trace(go.Bar(
        x=df_proy["Horizonte"], y=df_proy["Stock Disponible"],
        name="Stock Disponible", marker_color="#22c55e"
    ))
    fig.add_trace(go.Scatter(
        x=df_proy["Horizonte"], y=df_proy["Stock Recomendado"],
        mode="lines+markers", name="Stock Recomendado (+15%)",
        line=dict(color="#f59e0b", dash="dash", width=2)
    ))
    fig.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=30,b=0), legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla
    _section("📋 Detalle de proyección")
    st.dataframe(df_proy, use_container_width=True)

    return df_proy


# ══════════════════════════════════════════════════════════
# SECCIÓN: ALERTAS
# ══════════════════════════════════════════════════════════

def sec_alertas(df: pd.DataFrame, df_proy: pd.DataFrame) -> None:
    st.markdown("<h2 style='color:#60a5fa;'>🚨 Panel de Alertas Tempranas</h2>",
                unsafe_allow_html=True)

    col_qb = "Quiebre_Stock" if "Quiebre_Stock" in df.columns else None

    # Alertas históricas
    if col_qb:
        _section("📅 Quiebres históricos (últimos 30 días)")
        df_rec = df.copy()
        if "Fecha" in df_rec.columns:
            df_rec = df_rec.sort_values("Fecha").tail(200)
        df_qb = df_rec[df_rec[col_qb] == "Si"]

        if len(df_qb) == 0:
            st.success("✅ Sin quiebres de stock en el período seleccionado.")
        else:
            st.warning(f"⚠️ Se detectaron **{len(df_qb)} eventos** de quiebre de stock.")
            for _, row in df_qb.head(10).iterrows():
                fecha = row.get("Fecha", "—")
                cedi  = row.get("CEDI", "—")
                dem   = row.get("Demanda_Real", "—")
                st.markdown(
                    f"🔴 **{cedi}** | {fecha} | Demanda: **{dem}**",
                    unsafe_allow_html=True,
                )

    # Alertas proyectadas
    if not df_proy.empty:
        _section("🔮 Alertas proyectadas (72 horas)")
        for _, row in df_proy.iterrows():
            icono = {"Alto":"🔴","Medio":"🟡","Bajo":"🟢"}[row["Riesgo"]]
            msg   = f"{icono} **{row['Horizonte']}** — {row['CEDI']} — {row['Alerta']} (Demanda: {row['Demanda Predicha']:,})"
            if row["Riesgo"] == "Alto":
                st.error(msg)
            elif row["Riesgo"] == "Medio":
                st.warning(msg)
            else:
                st.success(msg)

    # Gauge SAT
    col_sat = next((c for c in ["Satisfaccion_SAT","Indice_Satisfaccion_SAT"] if c in df.columns), None)
    if col_sat:
        _section("😊 Índice de Satisfacción SAT")
        sat_val = float(df[col_sat].mean())
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sat_val,
            delta={"reference": 4.0},
            gauge={
                "axis": {"range": [1, 5]},
                "bar":  {"color": SUCCESS_COLOR},
                "steps": [
                    {"range": [1, 3], "color": "#7f1d1d"},
                    {"range": [3, 4], "color": "#78350f"},
                    {"range": [4, 5], "color": "#14532d"},
                ],
                "threshold": {"line": {"color": "white","width": 2}, "thickness": 0.75, "value": 4.0},
            },
            title={"text": "Promedio SAT"},
            number={"suffix": " / 5.0"},
        ))
        fig.update_layout(
            height=260, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20,r=20,t=40,b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# SECCIÓN: ANÁLISIS EXPLORATORIO
# ══════════════════════════════════════════════════════════

def sec_eda(df: pd.DataFrame, opts: dict) -> None:
    st.markdown("<h2 style='color:#60a5fa;'>📊 Análisis Exploratorio de Datos</h2>",
                unsafe_allow_html=True)
    df_f = _filtrar(df, opts)

    col1, col2 = st.columns(2)

    with col1:
        _section("🌡️ Temperatura vs Demanda")
        temp_col = "Temperatura_Promedio_C"
        dem_col  = "Demanda_Real"
        if temp_col in df_f.columns and dem_col in df_f.columns:
            fig = px.scatter(df_f, x=temp_col, y=dem_col, color="CEDI",
                             color_discrete_sequence=["#3b82f6","#f59e0b"],
                             template="plotly_dark", opacity=0.5, trendline="ols")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        _section("🕐 Demanda por día de la semana")
        if "Dia_Semana" in df_f.columns and "Demanda_Real" in df_f.columns:
            dias = {0:"Lun",1:"Mar",2:"Mié",3:"Jue",4:"Vie",5:"Sáb",6:"Dom"}
            df_ds = df_f.copy()
            df_ds["Día"] = df_ds["Dia_Semana"].map(dias)
            df_agg = df_ds.groupby(["Día","CEDI"])["Demanda_Real"].mean().reset_index()
            order  = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
            fig = px.bar(df_agg, x="Día", y="Demanda_Real", color="CEDI",
                         barmode="group", category_orders={"Día": order},
                         color_discrete_sequence=["#3b82f6","#f59e0b"],
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Correlación
    _section("🔗 Mapa de correlaciones")
    num_df = df_f.select_dtypes("number")
    if not num_df.empty:
        corr   = num_df.corr().round(2)
        fig_c  = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                           template="plotly_dark", aspect="auto")
        fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_c, use_container_width=True)

    # Stats descriptivos
    with st.expander("📐 Estadísticos descriptivos"):
        st.dataframe(df_f.describe().T.round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════
# SECCIÓN: ACERCA DE
# ══════════════════════════════════════════════════════════

def sec_about() -> None:
    st.markdown("<h2 style='color:#60a5fa;'>ℹ️ Acerca del Sistema</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    | Campo | Detalle |
    |---|---|
    | **Sistema** | {APP_NAME} v{APP_VERSION} |
    | **Autor** | {AUTHOR} |
    | **Institución** | {UNIVERSIDAD} |
    | **Curso** | Trabajo de Grado 3 — Modelos de Innovación |
    | **Modelo ML** | Regresión lineal, Gradient boosting, Árbol de decisión, Random Forest Regressor (Scikit-Learn) |
    | **Framework** | Streamlit ≥ 1.40 |
    | **Técnica innovación** | SCAMPER |
    | **Repositorio** | https://github.com/W-Andres/SATMLSCM3 |

    ### 🏗️ Arquitectura SAT-ML
    ```
    CSV / ERP ──► ETL (Pandas) ──► Feature Engineering
                                        │
                                 Train / Test Split (80/20)
                                        │
                        ┌───────────────┼───────────────┐
                   Lin. Reg.    Árbol Dec.   Grad. Boost.  Random Forest ⭐
                        └───────────────┼───────────────┘
                                  Comparación R² / MAPE
                                        │
                               Mejor Modelo (Pipeline)
                                        │
                            Proyección 72h + Alertas SAT-ML
                                        │
                               Dashboard Ejecutivo (Streamlit)
    ```

    ### 📐 Técnica SCAMPER aplicada
    - **S**ustituir: Excel / intuición → ML estadístico
    - **C**ombinar: logística (stock, tráfico) + marketing (SAT) = SAT-ML
    - **A**daptar: Python OSS para CEDIs sin licencias costosas
    - **M**odificar: tablas densas → alertas semafóricas
    - **P**roponer usos: planificación de personal en picos
    - **E**liminar: latencia semanal → análisis en segundos
    - **R**eorganizar: reactivo → predictivo (anticipa el quiebre)
    """)


# ══════════════════════════════════════════════════════════
# EXPORTACIONES (barra inferior)
# ══════════════════════════════════════════════════════════

def render_exports(df: pd.DataFrame, comp_df: pd.DataFrame,
                   df_proy: pd.DataFrame, best_name: str) -> None:
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    _section("💾 Exportar resultados")
    col1, col2, col3 = st.columns(3)

    with col1:
        df_exp = df_proy if not df_proy.empty else df.head(100)
        st.download_button(
            "📥 Descargar Excel",
            data=exportar_excel(df, comp_df if comp_df is not None else pd.DataFrame(), df_exp),
            file_name="satml_reporte.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col2:
        if comp_df is not None and not df_proy.empty:
            st.download_button(
                "📄 Descargar PDF",
                data=exportar_pdf(comp_df, df_proy, best_name or "—"),
                file_name="satml_informe.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with col3:
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📋 Descargar CSV",
            data=csv_bytes,
            file_name="historico_satml.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════
# FILTRADO
# ══════════════════════════════════════════════════════════

def _filtrar(df: pd.DataFrame, opts: dict) -> pd.DataFrame:
    df_f = df.copy()

    if opts["cedi"] != "Todos":
        df_f = df_f[df_f["CEDI"] == opts["cedi"]]

    cat_col = opts.get("cat_col")
    if cat_col and opts["cat"] != "Todas":
        df_f = df_f[df_f[cat_col] == opts["cat"]]

    if opts["rango"] and "Fecha" in df_f.columns and len(opts["rango"]) == 2:
        start, end = opts["rango"]
        df_f = df_f[
            (df_f["Fecha"].dt.date >= start) &
            (df_f["Fecha"].dt.date <= end)
        ]

    return df_f


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main() -> None:

    # ── Autenticación ──
    if not st.session_state.get("authenticated", False):
        login()
        st.stop()

    # ── Carga de datos ──
    df = cargar_datos(st.session_state.get("uploaded_file"))

    # ── Sidebar ──
    opts = render_sidebar(df)

    # Si se sube un archivo nuevo, lo guardamos en session_state y recargamos
    if opts["uploaded"] is not None:
        st.session_state["uploaded_file"] = opts["uploaded"]
        st.cache_data.clear()
        st.rerun()

    # Estado de sesión para el modelo entrenado
    if "best_pipe" not in st.session_state:
        st.session_state["best_pipe"]  = None
        st.session_state["comp_df"]    = None
        st.session_state["best_name"]  = None
        st.session_state["df_proy"]    = pd.DataFrame()

    # ── Routing de secciones ──
    sec = opts["seccion"]

    if sec == "🏠 Dashboard Ejecutivo":
        sec_dashboard(df, opts)

    elif sec == "🤖 Entrenamiento":
        pipe, comp, name = sec_entrenamiento(df)
        if pipe is not None:
            st.session_state["best_pipe"]  = pipe
            st.session_state["comp_df"]    = comp
            st.session_state["best_name"]  = name

    elif sec == "📈 Predicción 72h":
        df_proy = sec_prediccion(
            df,
            st.session_state["best_pipe"],
            st.session_state["best_name"] or "—",
            opts,
        )
        st.session_state["df_proy"] = df_proy

    elif sec == "🚨 Alertas":
        sec_alertas(df, st.session_state["df_proy"])

    elif sec == "📊 Análisis Exploratorio":
        sec_eda(df, opts)

    elif sec == "ℹ️ Acerca de":
        sec_about()

    # ── Exportaciones ──
    render_exports(
        df,
        st.session_state["comp_df"],
        st.session_state["df_proy"],
        st.session_state["best_name"] or "—",
    )

    # ── Footer ──
    st.markdown(f'<div class="footer">{FOOTER}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
