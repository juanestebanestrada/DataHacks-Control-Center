# DataHacks Control Center
# Centro de control para análisis de fútbol profesional

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Debe ir antes de importar pyplot
import matplotlib.pyplot as plt

# Configuración de página con tema premium
st.set_page_config(
    page_title="DataHacks Control Center",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para diseño premium - Tema Negro con Tiffany Blue (1837 Blue)
st.markdown("""
<style>
    /* Tema negro con acentos Tiffany Blue #0ABAB5 */
    .stApp {
        background: #0a0a0a;
    }
    
    /* Cards con glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(10, 186, 181, 0.3);
        margin: 10px 0;
    }
    
    /* Botones con gradiente Tiffany Blue */
    .stButton > button {
        background: linear-gradient(90deg, #089992 0%, #0ABAB5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(10, 186, 181, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(10, 186, 181, 0.6);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        border: 1px solid rgba(10, 186, 181, 0.3);
    }
    
    /* Headers con gradiente Tiffany Blue */
    h1 {
        background: linear-gradient(90deg, #089992, #0ABAB5, #5DD9D5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #0a0a0a;
    }
    
    /* DataFrame styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(10, 186, 181, 0.05);
        border-radius: 12px;
        padding: 8px;
        border: 1px solid rgba(10, 186, 181, 0.15);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #0ABAB5;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #089992, #0ABAB5);
        color: white;
    }
    
    /* Spinner animation */
    .loading-spinner {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(10, 186, 181, 0.2);
        border-left: 4px solid #0ABAB5;
        border-radius: 8px;
    }
    
    .stError {
        background: rgba(220, 53, 69, 0.2);
        border-left: 4px solid #dc3545;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado de sesión
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'current_source' not in st.session_state:
    st.session_state.current_source = None
if 'gpu_available' not in st.session_state:
    # Detectar GPU
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        st.session_state.gpu_available = True
    except:
        st.session_state.gpu_available = False

# Header principal
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("# ⚽ DataHacks Control Center")
    st.markdown("*Centro de control para análisis de fútbol profesional*")

# Indicador de GPU
with col3:
    if st.session_state.gpu_available:
        st.markdown("🟢 **GPU CUDA Activa**")
    else:
        st.markdown("🟡 **Modo CPU**")

st.markdown("---")

# Navegación por tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Extracción de Datos", "📈 Análisis Avanzado", "🎯 Simulación Poisson", "🎨 Visualizaciones", "ℹ️ Información"])

# ============================================
# TAB 1: EXTRACCIÓN DE DATOS
# ============================================
with tab1:
    from utils.data_sources import (
        get_sources, 
        get_leagues_for_source, 
        get_seasons_for_league,
        get_stats_types,
        extract_data
    )
    
    st.markdown("### 🔍 Configuración de Extracción")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source = st.selectbox(
            "📌 Fuente de Datos",
            options=get_sources(),
            help="Selecciona la página web de estadísticas"
        )
    
    with col2:
        leagues = get_leagues_for_source(source)
        league = st.selectbox(
            "🏆 Liga",
            options=leagues,
            help="Ligas disponibles para esta fuente"
        )
    
    with col3:
        seasons = get_seasons_for_league(source, league)
        season = st.selectbox(
            "📅 Temporada",
            options=seasons if seasons else ["No disponible"],
            help="Temporadas disponibles"
        )
    
    # Opciones adicionales
    col1, col2 = st.columns(2)
    with col1:
        stat_types = get_stats_types(source)
        stat_type = st.selectbox(
            "📋 Tipo de Estadísticas",
            options=stat_types,
            help="Tipo de datos a extraer"
        )
    
    with col2:
        if source == "FBRef":
            fbref_stats = ['stats', 'shooting', 'passing', 'defense', 'gca', 'possession']
            stat_category = st.selectbox(
                "📊 Categoría (FBRef)",
                options=fbref_stats
            )
        else:
            stat_category = None
    
    st.markdown("---")
    
    # Botón de extracción
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        extract_button = st.button("🚀 Extraer Datos", use_container_width=True)
    
    if extract_button:
        with st.spinner("⏳ Extrayendo datos... Esto puede tomar unos segundos"):
            try:
                data = extract_data(
                    source=source,
                    league=league,
                    season=season,
                    stat_type=stat_type,
                    stat_category=stat_category
                )
                
                if data is not None and not data.empty:
                    st.session_state.extracted_data = data
                    st.session_state.current_source = source
                    st.success(f"✅ ¡Datos extraídos exitosamente! {len(data)} registros encontrados.")
                else:
                    st.warning("⚠️ No se encontraron datos para esta selección.")
            except Exception as e:
                st.error(f"❌ Error durante la extracción: {str(e)}")
    
    # Mostrar datos extraídos
    if st.session_state.extracted_data is not None:
        st.markdown("### 📊 Datos Extraídos")
        
        # Métricas resumen
        data = st.session_state.extracted_data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Registros", len(data))
        with col2:
            st.metric("📊 Columnas", len(data.columns))
        with col3:
            st.metric("📌 Fuente", st.session_state.current_source)
        with col4:
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("💾 Memoria", f"{memory_mb:.2f} MB")
        
        # Tabla interactiva
        st.dataframe(
            data,
            use_container_width=True,
            height=400
        )
        
        # Botones de exportación
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar CSV",
                data=csv,
                file_name=f"lanusstats_{source}_{league}.csv",
                mime="text/csv"
            )
        with col2:
            # Excel export
            import io
            buffer = io.BytesIO()
            data.to_excel(buffer, index=False, engine='openpyxl')
            st.download_button(
                "📥 Descargar Excel",
                data=buffer.getvalue(),
                file_name=f"lanusstats_{source}_{league}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ============================================
# TAB 2: ANÁLISIS AVANZADO
# ============================================
with tab2:
    st.markdown("### 📈 Análisis Avanzado con GPU")
    
    if st.session_state.extracted_data is None:
        st.info("ℹ️ Primero extrae datos en la pestaña 'Extracción de Datos' para poder analizarlos.")
    else:
        from utils.gpu_processor import GPUProcessor
        import plotly.express as px
        import plotly.graph_objects as go
        
        gpu = GPUProcessor()
        data = st.session_state.extracted_data
        
        # Estado de GPU
        if gpu.gpu_available:
            st.success("🎮 GPU NVIDIA detectada - Procesamiento acelerado activo")
        else:
            st.warning("⚠️ GPU no detectada - Usando CPU para cálculos")
        
        st.markdown("---")
        
        # Mejorar la detección de columnas numéricas
        # Intentar convertir columnas que parecen numéricas pero están como string
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    # Intentar convertir a numérico
                    converted = pd.to_numeric(data[col], errors='coerce')
                    # Si más del 50% de valores son numéricos válidos, usar la conversión
                    if converted.notna().sum() > len(data) * 0.5:
                        data[col] = converted
                except:
                    pass
        
        # Actualizar datos en sesión con las conversiones
        st.session_state.extracted_data = data
        
        # Ahora detectar columnas numéricas (incluyendo int32, float32, etc.)
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ No hay suficientes columnas numéricas para análisis")
            st.info(f"Columnas disponibles: {list(data.columns[:10])}... (mostrando primeras 10)")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Análisis de Distribución")
                selected_col = st.selectbox("Selecciona columna", numeric_cols)
                
                if selected_col:
                    # Estadísticas calculadas con GPU
                    stats = gpu.compute_statistics(data[selected_col].values)
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Media", f"{stats['mean']:.2f}")
                    col_b.metric("Std Dev", f"{stats['std']:.2f}")
                    col_c.metric("Mediana", f"{stats['median']:.2f}")
                    
                    # Histograma con Plotly
                    fig = px.histogram(
                        data, x=selected_col,
                        title=f"Distribución de {selected_col}",
                        template="plotly_dark",
                        color_discrete_sequence=['#e94560']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Comparación de Métricas")
                
                x_col = st.selectbox("Eje X", numeric_cols, key="x_axis")
                y_col = st.selectbox("Eje Y", numeric_cols, index=min(1, len(numeric_cols)-1), key="y_axis")
                
                # Scatter plot
                fig = px.scatter(
                    data, x=x_col, y=y_col,
                    title=f"{x_col} vs {y_col}",
                    template="plotly_dark",
                    color_discrete_sequence=['#ff6b6b'],
                    trendline="ols" if len(data) > 5 else None
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Radar chart para comparación
            st.markdown("#### 🎯 Comparación Radar (Top 5)")
            
            if 'Player' in data.columns or 'Squad' in data.columns or 'Team' in data.columns:
                name_col = next((c for c in ['Player', 'Squad', 'Team', 'player', 'team'] if c in data.columns), None)
                
                if name_col:
                    selected_metrics = st.multiselect(
                        "Selecciona métricas para el radar",
                        numeric_cols,
                        default=numeric_cols[:min(5, len(numeric_cols))]
                    )
                    
                    if len(selected_metrics) >= 3:
                        # Normalizar datos para radar con GPU
                        normalized = gpu.normalize_for_radar(data, selected_metrics)
                        top_5 = normalized.head(5)
                        
                        fig = go.Figure()
                        
                        colors = ['#e94560', '#ff6b6b', '#ffc371', '#2ecc71', '#3498db']
                        for idx, (_, row) in enumerate(top_5.iterrows()):
                            fig.add_trace(go.Scatterpolar(
                                r=[row[m] for m in selected_metrics],
                                theta=selected_metrics,
                                fill='toself',
                                name=str(row[name_col])[:20],
                                line=dict(color=colors[idx % len(colors)])
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 1]),
                                bgcolor='rgba(0,0,0,0)'
                            ),
                            showlegend=True,
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            title="Comparación Top 5"
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3: SIMULACIÓN POISSON
# ============================================
with tab3:
    st.markdown("### 🎯 Simulación de Temporada con Poisson")
    
    if st.session_state.extracted_data is None:
        st.info("ℹ️ Primero extrae datos de **equipos** en la pestaña 'Extracción de Datos'.")
        st.markdown("""
        **Requisitos para simulación:**
        - Extraer datos de **FBRef** con tipo **Equipos**
        - Los datos deben incluir columnas de goles (GF/GA o Gls/GA)
        """)
    else:
        from utils.poisson_simulator import PoissonSimulator
        import plotly.express as px
        import plotly.graph_objects as go
        
        data = st.session_state.extracted_data
        
        st.markdown("""
        Este simulador usa la **distribución de Poisson** para proyectar resultados de la próxima temporada 
        basándose en las estadísticas históricas de goles marcados y recibidos.
        """)
        
        st.markdown("---")
        
        # Configuración
        col1, col2 = st.columns(2)
        
        with col1:
            num_sims = st.slider(
                "🔄 Número de simulaciones",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Más simulaciones = resultados más precisos pero más tiempo"
            )
        
        with col2:
            home_advantage = st.slider(
                "🏠 Ventaja local (%)",
                min_value=0,
                max_value=50,
                value=25,
                step=5,
                help="Porcentaje de ventaja para el equipo local"
            )
        
        # Detectar columnas disponibles
        available_cols = data.columns.tolist()
        team_col = None
        gf_col = None
        ga_col = None
        mp_col = None
        
        # Buscar columnas automáticamente (case-insensitive)
        for col in available_cols:
            col_lower = str(col).lower()
            if team_col is None and col_lower in ['squad', 'team', 'equipo']:
                team_col = col
            elif gf_col is None and col_lower in ['gf', 'gls', 'goals', 'goles']:
                gf_col = col
            elif ga_col is None and col_lower in ['ga', 'gc', 'goals against', 'goalsagainst']:
                ga_col = col
            elif mp_col is None and col_lower in ['mp', 'pj', 'matches', 'partidos']:
                mp_col = col
        
        # Verificar columnas requeridas y mostrar mensajes útiles
        missing_cols = []
        if not gf_col:
            missing_cols.append("Goles a Favor (Gls/GF)")
        if not ga_col:
            missing_cols.append("Goles en Contra (GA/GC)")
        if not mp_col:
            missing_cols.append("Partidos Jugados (MP)")
        
        if missing_cols:
            st.warning(f"⚠️ **Columnas faltantes:** {', '.join(missing_cols)}")
            st.info("""
            💡 **Para obtener todas las columnas necesarias:**
            1. Ve a la pestaña **'Extracción de Datos'**
            2. Selecciona **FBRef** como fuente
            3. Selecciona tipo: **'Equipos (Poisson)'** ← Importante!
            4. Extrae los datos nuevamente
            
            Este tipo especial combina tablas de shooting y keeper para incluir Gls, GA y MP.
            """)
            st.markdown(f"📋 **Columnas actuales:** `{available_cols}`")
        
        # Mostrar configuración de columnas
        with st.expander("⚙️ Configuración de columnas", expanded=bool(missing_cols)):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                team_col = st.selectbox("Columna Equipo", available_cols, 
                                        index=available_cols.index(team_col) if team_col in available_cols else 0)
            with col2:
                gf_col = st.selectbox("Columna Goles Favor", available_cols,
                                      index=available_cols.index(gf_col) if gf_col in available_cols else 0)
            with col3:
                ga_col = st.selectbox("Columna Goles Contra", available_cols,
                                      index=available_cols.index(ga_col) if ga_col in available_cols else 0)
            with col4:
                mp_col = st.selectbox("Columna Partidos", available_cols,
                                      index=available_cols.index(mp_col) if mp_col in available_cols else 0)
        
        st.markdown("---")
        
        # Botón de simulación
        if st.button("🚀 Ejecutar Simulación", use_container_width=True):
            try:
                # Crear simulador
                simulator = PoissonSimulator(home_advantage=1 + home_advantage/100)
                
                # Calcular fortalezas
                simulator.calculate_team_strengths(
                    data,
                    team_col=team_col,
                    goals_for_col=gf_col,
                    goals_against_col=ga_col,
                    matches_col=mp_col
                )
                
                # Ejecutar simulación con barra de progreso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(p):
                    progress_bar.progress(p)
                    status_text.text(f"Simulando temporada {int(p * num_sims)}/{num_sims}...")
                
                with st.spinner("⏳ Ejecutando simulación Monte Carlo..."):
                    results = simulator.simulate_season(
                        num_simulations=num_sims,
                        progress_callback=update_progress
                    )
                
                progress_bar.progress(1.0)
                status_text.text("✓ Simulación completada!")
                
                # Guardar resultados en sesión
                st.session_state.simulation_results = results
                
                st.success(f"✅ Simulación completada: {num_sims} temporadas simuladas")
                
            except Exception as e:
                st.error(f"❌ Error en simulación: {str(e)}")
                st.info("Asegúrate de usar datos de **Equipos** con columnas de goles (GF, GA)")
        
        # Mostrar resultados si existen
        if 'simulation_results' in st.session_state and st.session_state.simulation_results is not None:
            results = st.session_state.simulation_results
            
            st.markdown("### 📊 Tabla Proyectada")
            
            # Mostrar tabla estilizada
            st.dataframe(
                results,
                use_container_width=True,
                height=400
            )
            
            # Gráficos
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏆 Probabilidad de Campeón")
                fig = px.bar(
                    results.head(10),
                    x='Equipo',
                    y='% Campeón',
                    title="Probabilidad de ser Campeón (Top 10)",
                    template="plotly_dark",
                    color='% Campeón',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### ⚠️ Riesgo de Descenso")
                descenso_df = results.nlargest(10, '% Descenso')
                fig = px.bar(
                    descenso_df,
                    x='Equipo',
                    y='% Descenso',
                    title="Probabilidad de Descenso (Top 10)",
                    template="plotly_dark",
                    color='% Descenso',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Descargar resultados
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar Proyección CSV",
                data=csv,
                file_name="simulacion_poisson.csv",
                mime="text/csv"
            )

# ============================================
# TAB 4: VISUALIZACIONES (StatsBomb)
# ============================================
with tab4:
    st.markdown("### 🎨 Visualizaciones con StatsBomb")
    
    from utils.statsbomb_utils import (
        get_available_competitions,
        get_matches_for_competition,
        get_events_for_match,
        get_shots,
        get_passes,
        get_players_from_events,
        get_teams_from_events
    )
    from mplsoccer import Pitch, VerticalPitch
    
    st.markdown("""
    Explora datos de partidos utilizando los **datos abiertos de StatsBomb**.
    Puedes visualizar mapas de tiros, pases y heatmaps de equipos.
    """)
    
    st.markdown("---")
    
    # Inicializar TODAS las claves de session_state para StatsBomb ANTES de usarlas
    # Esto previene reruns cuando Streamlit inicializa nuevas claves
    defaults = {
        'sb_competitions_data': None,
        'sb_match_id': None,
        'sb_match_name': None,
        'statsbomb_events': None,
        'statsbomb_match': None,
        'statsbomb_comp': None,
        'statsbomb_season': None,
        'statsbomb_match': None,
        '_sb_initialized': False
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Marcar como inicializado para evitar futuros resets
    if not st.session_state._sb_initialized:
        st.session_state._sb_initialized = True
    
    # Cargar competiciones solo una vez al inicio
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_competitions():
        from statsbombpy import sb
        return sb.competitions()
    
    # Selección de competición
    st.markdown("#### 🏆 Seleccionar Partido")
    
    try:
        competitions = load_competitions()
        
        if competitions is not None and not competitions.empty:
            # Obtener lista de competiciones únicas
            unique_comps = competitions[['competition_id', 'competition_name']].drop_duplicates()
            comp_names = sorted(unique_comps['competition_name'].tolist())
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_comp = st.selectbox(
                    "📌 Competición",
                    options=comp_names,
                    key="statsbomb_comp"
                )
            
            with col2:
                if selected_comp:
                    comp_id = unique_comps[unique_comps['competition_name'] == selected_comp]['competition_id'].values[0]
                    seasons_df = competitions[competitions['competition_name'] == selected_comp][['season_id', 'season_name']].drop_duplicates()
                    season_names = seasons_df['season_name'].tolist()
                    
                    selected_season = st.selectbox(
                        "📅 Temporada",
                        options=season_names,
                        key="statsbomb_season"
                    )
            
            # Partidos
            if selected_comp and selected_season:
                season_id = seasons_df[seasons_df['season_name'] == selected_season]['season_id'].values[0]
                
                @st.cache_data(ttl=3600, show_spinner=False)
                def load_matches(cid, sid):
                    from statsbombpy import sb
                    return sb.matches(competition_id=cid, season_id=sid)
                
                with st.spinner("Cargando partidos..."):
                    matches = load_matches(int(comp_id), int(season_id))
                
                if matches is not None and not matches.empty:
                    matches = matches.copy()
                    matches['display'] = matches['home_team'] + ' vs ' + matches['away_team'] + ' (' + matches['match_date'].astype(str) + ')'
                    
                    selected_match_display = st.selectbox(
                        "⚽ Partido",
                        options=matches['display'].tolist(),
                        key="statsbomb_match"
                    )
                    
                    if selected_match_display:
                        match_id = matches[matches['display'] == selected_match_display]['match_id'].values[0]
                        
                        # Botón para cargar eventos
                        if st.button("🔄 Cargar Eventos del Partido", key="load_events_btn", use_container_width=True):
                            with st.spinner("⏳ Cargando eventos..."):
                                events = get_events_for_match(match_id)
                                if events is not None and not events.empty:
                                    st.session_state.statsbomb_events = events
                                    st.session_state.statsbomb_match = selected_match_display
                                    st.success(f"✅ {len(events)} eventos cargados")
                                else:
                                    st.warning("⚠️ No se encontraron eventos para este partido")
    except Exception as e:
        st.error(f"Error al cargar datos de StatsBomb: {str(e)}")
    
    st.markdown("---")
    
    # Visualizaciones si hay eventos cargados
    if 'statsbomb_events' in st.session_state and st.session_state.statsbomb_events is not None:
        events = st.session_state.statsbomb_events
        
        st.markdown(f"#### 📊 Visualizaciones: {st.session_state.statsbomb_match}")
        
        viz_type = st.radio(
            "Tipo de Visualización",
            ["🎯 Mapa de Tiros", "📍 Mapa de Pases", "🔥 Heatmap de Equipo", "👥 Comparar Jugadores"],
            horizontal=True
        )
        
        if viz_type == "🎯 Mapa de Tiros":
            st.markdown("##### Mapa de Tiros del Partido")
            
            shots = get_shots(events)
            
            if not shots.empty:
                teams = get_teams_from_events(shots)
                selected_team = st.selectbox("Filtrar por equipo (opcional)", ["Todos"] + teams)
                
                if selected_team != "Todos":
                    shots = shots[shots['team'] == selected_team]
                
                # Crear visualización
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor('#0a0a0a')
                ax.set_facecolor('#0a0a0a')
                
                pitch = Pitch(
                    pitch_type='statsbomb',
                    pitch_color='#0a0a0a',
                    line_color='#0ABAB5',
                    linewidth=2
                )
                pitch.draw(ax=ax)
                
                # Plotear tiros
                for idx, shot in shots.iterrows():
                    location = shot.get('location', [0, 0])
                    if location and len(location) >= 2:
                        x, y = location[0], location[1]
                        outcome = shot.get('shot_outcome', 'Unknown')
                        
                        if outcome == 'Goal':
                            color = '#00ff00'
                            size = 200
                        elif outcome == 'Saved':
                            color = '#ff6b6b'
                            size = 100
                        else:
                            color = '#ffffff'
                            size = 80
                        
                        ax.scatter(x, y, c=color, s=size, alpha=0.7, edgecolors='white', linewidth=1)
                
                ax.set_title(f'Mapa de Tiros - {st.session_state.statsbomb_match}', color='white', fontsize=14, fontweight='bold')
                
                # Leyenda
                ax.scatter([], [], c='#00ff00', s=100, label='Gol')
                ax.scatter([], [], c='#ff6b6b', s=100, label='Salvado')
                ax.scatter([], [], c='#ffffff', s=100, label='Otro')
                ax.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#0ABAB5', labelcolor='white')
                
                st.pyplot(fig)
                plt.close()
                
                # Estadísticas de tiros
                st.markdown("###### Resumen de Tiros")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Tiros", len(shots))
                col2.metric("Goles", len(shots[shots.get('shot_outcome', pd.Series()) == 'Goal']) if 'shot_outcome' in shots.columns else 0)
                col3.metric("A Puerta", len(shots[shots.get('shot_outcome', pd.Series()).isin(['Goal', 'Saved'])]) if 'shot_outcome' in shots.columns else 0)
                col4.metric("xG Total", f"{shots['shot_statsbomb_xg'].sum():.2f}" if 'shot_statsbomb_xg' in shots.columns else "N/A")
            else:
                st.info("No hay tiros en este partido")
        
        elif viz_type == "📍 Mapa de Pases":
            st.markdown("##### Mapa de Pases")
            
            teams = get_teams_from_events(events)
            selected_team = st.selectbox("Seleccionar equipo", teams)
            
            if selected_team:
                team_events = events[events['team'] == selected_team]
                players = get_players_from_events(team_events)
                
                selected_player = st.selectbox("Seleccionar jugador", players)
                
                if selected_player:
                    passes = get_passes(events, player_name=selected_player)
                    
                    if not passes.empty:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        fig.patch.set_facecolor('#0a0a0a')
                        ax.set_facecolor('#0a0a0a')
                        
                        pitch = Pitch(
                            pitch_type='statsbomb',
                            pitch_color='#0a0a0a',
                            line_color='#0ABAB5',
                            linewidth=2
                        )
                        pitch.draw(ax=ax)
                        
                        # Plotear pases
                        for idx, pass_event in passes.iterrows():
                            location = pass_event.get('location', [0, 0])
                            end_location = pass_event.get('pass_end_location', [0, 0])
                            
                            if location and end_location and len(location) >= 2 and len(end_location) >= 2:
                                x_start, y_start = location[0], location[1]
                                x_end, y_end = end_location[0], end_location[1]
                                
                                outcome = pass_event.get('pass_outcome', None)
                                color = '#ff6b6b' if outcome else '#0ABAB5'
                                
                                ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))
                        
                        ax.set_title(f'Mapa de Pases - {selected_player}', color='white', fontsize=14, fontweight='bold')
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Estadísticas
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Pases", len(passes))
                        completed = len(passes[passes.get('pass_outcome', pd.Series()).isna()]) if 'pass_outcome' in passes.columns else len(passes)
                        col2.metric("Completados", completed)
                        col3.metric("Precisión", f"{(completed/len(passes)*100):.1f}%" if len(passes) > 0 else "N/A")
                    else:
                        st.info(f"No hay pases registrados para {selected_player}")
        
        elif viz_type == "🔥 Heatmap de Equipo":
            st.markdown("##### Heatmap de Actividad")
            
            teams = get_teams_from_events(events)
            selected_team = st.selectbox("Seleccionar equipo", teams, key="heatmap_team")
            
            if selected_team:
                team_events = events[events['team'] == selected_team]
                
                # Extraer ubicaciones
                locations = []
                for loc in team_events['location'].dropna():
                    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                        locations.append(loc)
                
                if locations:
                    import numpy as np
                    x_coords = [loc[0] for loc in locations]
                    y_coords = [loc[1] for loc in locations]
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    fig.patch.set_facecolor('#0a0a0a')
                    ax.set_facecolor('#0a0a0a')
                    
                    pitch = Pitch(
                        pitch_type='statsbomb',
                        pitch_color='#0a0a0a',
                        line_color='#0ABAB5',
                        linewidth=2
                    )
                    pitch.draw(ax=ax)
                    
                    # Crear heatmap
                    pitch.kdeplot(x_coords, y_coords, ax=ax, cmap='hot', fill=True, alpha=0.6, levels=50)
                    
                    ax.set_title(f'Heatmap de Actividad - {selected_team}', color='white', fontsize=14, fontweight='bold')
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    st.metric("Total Acciones", len(locations))
                else:
                    st.info("No hay datos de ubicación para este equipo")
        
        elif viz_type == "👥 Comparar Jugadores":
            st.markdown("##### Comparación de Jugadores")
            st.markdown("Selecciona dos jugadores del partido para comparar sus estadísticas.")
            
            # Obtener todos los jugadores del partido
            all_players = get_players_from_events(events)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("###### 🔵 Jugador 1")
                player1 = st.selectbox("Seleccionar jugador", all_players, key="player1_compare")
            
            with col2:
                st.markdown("###### 🔴 Jugador 2")
                # Filtrar para no mostrar el mismo jugador
                players_for_2 = [p for p in all_players if p != player1]
                player2 = st.selectbox("Seleccionar jugador", players_for_2, key="player2_compare")
            
            if player1 and player2:
                # Obtener estadísticas de cada jugador
                p1_events = events[events['player'] == player1]
                p2_events = events[events['player'] == player2]
                
                # Calcular estadísticas
                def get_player_stats(player_events):
                    stats = {}
                    stats['total_acciones'] = len(player_events)
                    stats['tiros'] = len(player_events[player_events['type'] == 'Shot'])
                    stats['pases'] = len(player_events[player_events['type'] == 'Pass'])
                    
                    # Pases completados
                    passes = player_events[player_events['type'] == 'Pass']
                    if not passes.empty and 'pass_outcome' in passes.columns:
                        stats['pases_completados'] = len(passes[passes['pass_outcome'].isna()])
                    else:
                        stats['pases_completados'] = stats['pases']
                    
                    stats['regates'] = len(player_events[player_events['type'] == 'Dribble'])
                    stats['recuperaciones'] = len(player_events[player_events['type'] == 'Ball Recovery'])
                    stats['duelos'] = len(player_events[player_events['type'] == 'Duel'])
                    
                    # Goles
                    shots = player_events[player_events['type'] == 'Shot']
                    if not shots.empty and 'shot_outcome' in shots.columns:
                        stats['goles'] = len(shots[shots['shot_outcome'] == 'Goal'])
                    else:
                        stats['goles'] = 0
                    
                    # xG
                    if not shots.empty and 'shot_statsbomb_xg' in shots.columns:
                        stats['xg'] = shots['shot_statsbomb_xg'].sum()
                    else:
                        stats['xg'] = 0
                    
                    return stats
                
                stats1 = get_player_stats(p1_events)
                stats2 = get_player_stats(p2_events)
                
                st.markdown("---")
                st.markdown("##### 📊 Estadísticas Comparativas")
                
                # Tabla comparativa
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    st.markdown(f"**{player1}**")
                with col2:
                    st.markdown("**Estadística**")
                with col3:
                    st.markdown(f"**{player2}**")
                
                metrics = [
                    ('Total Acciones', 'total_acciones'),
                    ('Tiros', 'tiros'),
                    ('Goles', 'goles'),
                    ('xG', 'xg'),
                    ('Pases', 'pases'),
                    ('Pases Completados', 'pases_completados'),
                    ('Regates', 'regates'),
                    ('Recuperaciones', 'recuperaciones'),
                    ('Duelos', 'duelos')
                ]
                
                for metric_name, metric_key in metrics:
                    col1, col2, col3 = st.columns([2, 1, 2])
                    val1 = stats1[metric_key]
                    val2 = stats2[metric_key]
                    
                    # Formatear valores
                    if metric_key == 'xg':
                        val1_str = f"{val1:.2f}"
                        val2_str = f"{val2:.2f}"
                    else:
                        val1_str = str(val1)
                        val2_str = str(val2)
                    
                    # Indicador de quién tiene más
                    if val1 > val2:
                        val1_str = f"🟢 {val1_str}"
                    elif val2 > val1:
                        val2_str = f"🟢 {val2_str}"
                    
                    with col1:
                        st.write(val1_str)
                    with col2:
                        st.write(f"*{metric_name}*")
                    with col3:
                        st.write(val2_str)
                
                st.markdown("---")
                
                # Selector de tipo de visualización
                st.markdown("##### 🗺️ Mapa Comparativo en Cancha")
                
                action_type = st.selectbox(
                    "Selecciona qué acciones visualizar",
                    ["Tiros (Shots)", "Pases (Passes)", "Todas las Acciones", "Regates (Dribbles)", "Recuperaciones"],
                    key="compare_action_type"
                )
                
                # Mapear selección a tipo de evento
                action_map = {
                    "Tiros (Shots)": "Shot",
                    "Pases (Passes)": "Pass",
                    "Regates (Dribbles)": "Dribble",
                    "Recuperaciones": "Ball Recovery",
                    "Todas las Acciones": None
                }
                
                selected_action = action_map[action_type]
                
                # Filtrar eventos según selección
                if selected_action:
                    p1_actions = p1_events[p1_events['type'] == selected_action]
                    p2_actions = p2_events[p2_events['type'] == selected_action]
                else:
                    p1_actions = p1_events[p1_events['location'].notna()]
                    p2_actions = p2_events[p2_events['location'].notna()]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                fig.patch.set_facecolor('#0a0a0a')
                
                for ax, actions_df, player_name, color in [(ax1, p1_actions, player1, '#3498db'), (ax2, p2_actions, player2, '#e74c3c')]:
                    ax.set_facecolor('#0a0a0a')
                    
                    pitch = Pitch(
                        pitch_type='statsbomb',
                        pitch_color='#0a0a0a',
                        line_color='#0ABAB5',
                        linewidth=2
                    )
                    pitch.draw(ax=ax)
                    
                    if not actions_df.empty:
                        for idx, action in actions_df.iterrows():
                            location = action.get('location', None)
                            if location is not None and isinstance(location, (list, tuple)) and len(location) >= 2:
                                x, y = location[0], location[1]
                                
                                # Color según tipo de acción o resultado
                                if selected_action == 'Shot':
                                    outcome = action.get('shot_outcome', 'Unknown')
                                    if outcome == 'Goal':
                                        marker_color = '#00ff00'
                                        size = 250
                                    else:
                                        marker_color = color
                                        size = 120
                                elif selected_action == 'Pass':
                                    end_loc = action.get('pass_end_location', None)
                                    outcome = action.get('pass_outcome', None)
                                    marker_color = '#ff6b6b' if outcome else color
                                    size = 60
                                    # Dibujar flecha para pases
                                    if end_loc and isinstance(end_loc, (list, tuple)) and len(end_loc) >= 2:
                                        ax.annotate('', xy=(end_loc[0], end_loc[1]), xytext=(x, y),
                                                   arrowprops=dict(arrowstyle='->', color=marker_color, lw=1, alpha=0.5))
                                else:
                                    marker_color = color
                                    size = 80
                                
                                ax.scatter(x, y, c=marker_color, s=size, alpha=0.7, edgecolors='white', linewidth=1)
                        
                        action_label = action_type.split(" ")[0].lower()
                        ax.set_title(f'{player_name}\n({len(actions_df)} {action_label})', color='white', fontsize=12, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Leyenda explicativa
                    if selected_action == 'Shot':
                        st.caption("🟢 Verde = Gol | Color del jugador = Otros tiros")
                    elif selected_action == 'Pass':
                        st.caption("Las flechas muestran la dirección del pase. Rojo = pase fallido")
    else:
        st.warning("⚠️ No se pudieron cargar las competiciones de StatsBomb")

# ============================================
# TAB 5: INFORMACIÓN
# ============================================
with tab5:
    st.markdown("### ℹ️ Acerca de DataHacks Control Center")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔗 Fuentes Soportadas
        
        | Fuente | Descripción |
        |--------|-------------|
        | **FBRef** | Estadísticas detalladas de ligas europeas y americanas |
        | **FotMob** | Rankings de jugadores y equipos por temporada |
        | **SofaScore** | Datos de partidos y jugadores |
        | **365Scores** | Estadísticas en tiempo real |
        | **Transfermarkt** | Valoraciones de mercado |
        | **StatsBomb** | Datos abiertos detallados de eventos |
        """)
    
    with col2:
        st.markdown("""
        #### ⚡ Características GPU
        
        - Procesamiento paralelo con CUDA
        - Cálculos estadísticos acelerados
        - Normalización de datos en GPU
        - Fallback automático a CPU
        
        #### 📦 Tecnologías
        - LanusStats v1.7.2
        - Streamlit
        - Plotly
        - CuPy (GPU)
        - Simulación Poisson
        - StatsBomb + mplsoccer
        """)
    
    st.markdown("---")
    st.markdown("*DataHacks Control Center - Análisis de fútbol con ❤️*")
