# Utilidades para gestión de fuentes de datos LanusStats

from LanusStats import Fbref, FotMob, SofaScore, ThreeSixFiveScores
from LanusStats.transfermarkt import Transfermarkt
from LanusStats.functions import get_available_leagues, get_available_season_for_leagues
import pandas as pd
import time
import logging
from functools import wraps

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('LanusStats')

# Configuración de reintentos
RETRY_CONFIG = {
    "FBRef": {"max_retries": 3, "base_delay": 2, "timeout": 30},
    "FotMob": {"max_retries": 3, "base_delay": 1, "timeout": 20},
    "SofaScore": {"max_retries": 3, "base_delay": 3, "timeout": 60},
    "365Scores": {"max_retries": 2, "base_delay": 1, "timeout": 15},
    "Transfermarkt": {"max_retries": 3, "base_delay": 2, "timeout": 30},
}


def retry_with_backoff(source: str):
    """
    Decorador para reintentos con backoff exponencial.
    
    Args:
        source: Nombre de la fuente de datos para obtener configuración
    """
    config = RETRY_CONFIG.get(source, {"max_retries": 3, "base_delay": 2, "timeout": 30})
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            max_retries = config["max_retries"]
            base_delay = config["base_delay"]
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"[{source}] Intento {attempt + 1}/{max_retries}")
                    result = func(*args, **kwargs)
                    logger.info(f"[{source}] ✓ Extracción exitosa")
                    return result
                    
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Errores que no vale la pena reintentar
                    if any(err in error_msg for err in ['not found', 'invalid', '404']):
                        logger.error(f"[{source}] Error no recuperable: {e}")
                        raise
                    
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Backoff exponencial
                        logger.warning(f"[{source}] Error: {e}. Reintentando en {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"[{source}] Falló después de {max_retries} intentos")
            
            raise last_exception
        return wrapper
    return decorator


def extract_sofascore_with_retry(league: str, season: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Extrae datos de SofaScore usando scraper personalizado con sesión persistente.
    
    El scraper original de LanusStats crea/destruye sesiones del navegador para cada
    solicitud, causando errores de "window already closed". Este wrapper usa un 
    scraper personalizado que mantiene una única sesión durante toda la extracción.
    
    Args:
        league: Liga a consultar
        season: Temporada
        max_retries: Número máximo de reintentos (pasado al scraper interno)
    
    Returns:
        DataFrame con los datos extraídos
    """
    # Usar el scraper personalizado con sesión persistente
    from utils.sofascore_scraper import SofaScoreScraper
    
    scraper = SofaScoreScraper()
    return scraper.scrape_league_stats(league=league, season=season)

# Mapeo de fuentes a clases
SOURCE_CLASSES = {
    "FBRef": Fbref,
    "FotMob": FotMob,
    "SofaScore": SofaScore,
    "365Scores": ThreeSixFiveScores,
    "Transfermarkt": Transfermarkt
}

# Mapeo de nombres internos a nombres de página
PAGE_NAMES = {
    "FBRef": "Fbref",
    "FotMob": "Fotmob",
    "SofaScore": "Sofascore",
    "365Scores": "365Scores",
    "Transfermarkt": "Transfermarkt"
}


def get_sources() -> list:
    """Obtiene las fuentes de datos disponibles."""
    return list(SOURCE_CLASSES.keys())


def get_leagues_for_source(source: str) -> list:
    """Obtiene las ligas disponibles para una fuente específica."""
    page_name = PAGE_NAMES.get(source, source)
    try:
        leagues = get_available_leagues(page_name)
        return leagues if leagues else ["No hay ligas disponibles"]
    except Exception:
        return ["No hay ligas disponibles"]


def get_seasons_for_league(source: str, league: str) -> list:
    """Obtiene las temporadas disponibles para una liga específica."""
    page_name = PAGE_NAMES.get(source, source)
    try:
        league_data = get_available_season_for_leagues(page_name, league)
        if league_data and 'seasons' in league_data and league_data['seasons']:
            seasons = list(league_data['seasons'].keys()) if isinstance(league_data['seasons'], dict) else list(league_data['seasons'])
            return sorted(seasons, reverse=True)
        return None
    except Exception:
        return None


def get_stats_types(source: str) -> list:
    """Obtiene los tipos de estadísticas disponibles por fuente."""
    stats_map = {
        "FBRef": ["Equipos", "Jugadores", "Equipos (Poisson)"],
        "FotMob": ["Equipos", "Jugadores", "Tablas"],
        "SofaScore": ["Jugadores Liga"],
        "365Scores": ["Top Jugadores"],
        "Transfermarkt": ["Valuaciones Equipos"]
    }
    return stats_map.get(source, ["General"])


def combine_fbref_for_poisson(league: str, season: str) -> pd.DataFrame:
    """
    Combina tablas de shooting (Gls) y keeper (GA) de FBRef para simulación Poisson.
    
    Args:
        league: Liga a consultar
        season: Temporada
        
    Returns:
        DataFrame con Squad, Gls (goles a favor), GA (goles en contra), MP (partidos)
    """
    from LanusStats import Fbref
    
    scraper = Fbref()
    logger.info(f"[FBRef Poisson] Obteniendo datos para {league} ({season})")
    
    shooting_df = None
    keeper_df = None
    
    # Intentar obtener tabla de shooting
    for stat_name in ['shooting', 'stats', 'standard']:
        try:
            logger.info(f"[FBRef Poisson] Intentando stat='{stat_name}'...")
            df = scraper.get_teams_season_stats(stat=stat_name, league=league, season=season)
            if df is not None:
                # Manejar tupla
                if isinstance(df, (list, tuple)):
                    df = df[0] if len(df) > 0 else None
                if df is not None and len(df) > 0:
                    shooting_df = fix_duplicate_columns(df)
                    logger.info(f"[FBRef Poisson] Obtenido '{stat_name}': {len(shooting_df)} equipos, columnas: {shooting_df.columns.tolist()[:10]}")
                    break
        except TypeError as te:
            logger.warning(f"[FBRef Poisson] TypeError en '{stat_name}': {te}")
            continue
        except Exception as e:
            logger.warning(f"[FBRef Poisson] Error en '{stat_name}': {e}")
            continue
    
    # Intentar obtener tabla de keeper para GA
    try:
        logger.info(f"[FBRef Poisson] Intentando stat='keeper' para GA...")
        df = scraper.get_teams_season_stats(stat='keeper', league=league, season=season)
        if df is not None:
            if isinstance(df, (list, tuple)):
                df = df[0] if len(df) > 0 else None
            if df is not None and len(df) > 0:
                keeper_df = fix_duplicate_columns(df)
                logger.info(f"[FBRef Poisson] Obtenido 'keeper': {len(keeper_df)} equipos, columnas: {keeper_df.columns.tolist()[:10]}")
    except TypeError as te:
        logger.warning(f"[FBRef Poisson] TypeError en 'keeper': {te}")
    except Exception as e:
        logger.warning(f"[FBRef Poisson] Error en 'keeper': {e}")
    
    # Si no tenemos datos de shooting, fallar
    if shooting_df is None or len(shooting_df) == 0:
        raise ValueError("No se pudieron obtener datos de FBRef. Verifica liga/temporada.")
    
    # Construir resultado con columnas garantizadas
    result = pd.DataFrame()
    
    # Columna de equipo
    squad_col = _find_column(shooting_df, ['Squad', 'squad', 'Team', 'team', 'Equipo'])
    if squad_col:
        result['Squad'] = shooting_df[squad_col].astype(str).str.strip()
    else:
        result['Squad'] = [f"Equipo_{i+1}" for i in range(len(shooting_df))]
    
    # Columna de goles (Gls)
    gls_col = _find_column(shooting_df, ['Gls', 'gls', 'Goals', 'goals', 'GF', 'G'])
    if gls_col:
        result['Gls'] = pd.to_numeric(shooting_df[gls_col], errors='coerce').fillna(0).astype(int)
        logger.info(f"[FBRef Poisson] Gls encontrado en columna '{gls_col}'")
    else:
        logger.warning("[FBRef Poisson] No se encontró columna de goles. Usando valores estimados.")
        result['Gls'] = 20  # Valor por defecto
    
    # Columna de partidos (MP)
    mp_col = _find_column(shooting_df, ['MP', 'mp', 'Matches', 'PJ', 'Games'])
    if mp_col:
        result['MP'] = pd.to_numeric(shooting_df[mp_col], errors='coerce').fillna(1).astype(int)
        logger.info(f"[FBRef Poisson] MP encontrado en columna '{mp_col}'")
    else:
        # Usar 90s si está disponible
        s90_col = _find_column(shooting_df, ['90s', '90'])
        if s90_col:
            result['MP'] = pd.to_numeric(shooting_df[s90_col], errors='coerce').fillna(1).round().astype(int)
            logger.info(f"[FBRef Poisson] MP calculado desde '90s'")
        else:
            result['MP'] = 10  # Valor por defecto
            logger.warning("[FBRef Poisson] No se encontró columna de partidos. Usando valor por defecto.")
    
    # Columna de goles en contra (GA)
    ga_found = False
    if keeper_df is not None and len(keeper_df) > 0:
        squad_col_k = _find_column(keeper_df, ['Squad', 'squad', 'Team', 'team', 'Equipo'])
        ga_col = _find_column(keeper_df, ['GA', 'ga', 'Goals Against', 'GoalsAgainst', 'GC', 'Goles en contra'])
        
        if squad_col_k and ga_col:
            keeper_df = keeper_df.copy()
            keeper_df['_squad_clean'] = keeper_df[squad_col_k].astype(str).str.strip()
            keeper_df['_ga'] = pd.to_numeric(keeper_df[ga_col], errors='coerce').fillna(0)
            ga_map = dict(zip(keeper_df['_squad_clean'], keeper_df['_ga']))
            result['GA'] = result['Squad'].map(ga_map)
            # Llenar NaN con promedio
            mean_ga = result['GA'].mean()
            if pd.isna(mean_ga):
                mean_ga = result['Gls'].mean()
            result['GA'] = result['GA'].fillna(mean_ga).astype(int)
            ga_found = True
            logger.info(f"[FBRef Poisson] GA encontrado en columna '{ga_col}'")
    
    if not ga_found:
        # Estimar GA como promedio de goles de la liga
        avg_gls = result['Gls'].mean()
        result['GA'] = int(avg_gls) if not pd.isna(avg_gls) else 15
        logger.warning(f"[FBRef Poisson] GA no encontrado. Estimado como promedio: {result['GA'].iloc[0]}")
    
    # Asegurar tipos correctos
    result['Gls'] = result['Gls'].astype(int)
    result['GA'] = result['GA'].astype(int)
    result['MP'] = result['MP'].astype(int)
    
    logger.info(f"[FBRef Poisson] ✓ Resultado final: {len(result)} equipos, columnas: {result.columns.tolist()}")
    return result


def _find_column(df: pd.DataFrame, candidates: list) -> str:
    """Busca una columna en el DataFrame probando múltiples nombres."""
    for col in candidates:
        if col in df.columns:
            return col
        # Buscar por coincidencia parcial
        for df_col in df.columns:
            if col.lower() in str(df_col).lower():
                return df_col
    return None


def extract_data(source: str, league: str, season: str, stat_type: str, stat_category: str = None) -> pd.DataFrame:
    """
    Extrae datos de la fuente especificada con reintentos automáticos.
    
    Args:
        source: Fuente de datos (FBRef, FotMob, etc.)
        league: Liga a consultar
        season: Temporada
        stat_type: Tipo de estadísticas (Equipos, Jugadores, etc.)
        stat_category: Categoría específica para FBRef
    
    Returns:
        DataFrame con los datos extraídos
    """
    scraper_class = SOURCE_CLASSES.get(source)
    if not scraper_class:
        raise ValueError(f"Fuente no soportada: {source}")
    
    logger.info(f"Iniciando extracción: {source} → {league} ({season})")
    
    try:
        if source == "FBRef":
            # Tipo especial para Poisson que combina shooting + keeper
            if stat_type == "Equipos (Poisson)":
                data = combine_fbref_for_poisson(league, season)
            else:
                data = _extract_fbref_with_retry(league, season, stat_type, stat_category)
        
        elif source == "FotMob":
            data = _extract_fotmob(league, season, stat_type)
        
        elif source == "SofaScore":
            data = extract_sofascore_with_retry(league=league, season=season)
        
        elif source == "365Scores":
            scraper = scraper_class()
            data = scraper.get_league_top_players_stats(league=league)
        
        elif source == "Transfermarkt":
            scraper = scraper_class()
            data = scraper.get_league_teams_valuations(league=league, season=season or "2024")
        
        else:
            raise ValueError(f"Extractor no implementado para: {source}")
        
        # Asegurar que devolvemos un DataFrame
        if isinstance(data, tuple):
            data = data[0]
        
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        # Manejar columnas duplicadas (común en FBRef con tablas GCA/SCA)
        data = fix_duplicate_columns(data)
        
        logger.info(f"✓ Extracción completada: {len(data)} registros")
        return data
    
    except Exception as e:
        logger.error(f"✗ Error en extracción: {str(e)}")
        raise Exception(f"Error extrayendo datos de {source}: {str(e)}")


def _extract_fbref_with_retry(league: str, season: str, stat_type: str, stat_category: str = None) -> pd.DataFrame:
    """Extrae datos de FBRef con reintentos automáticos."""
    config = RETRY_CONFIG["FBRef"]
    last_exception = None
    
    for attempt in range(config["max_retries"]):
        try:
            logger.info(f"[FBRef] Intento {attempt + 1}/{config['max_retries']} - {stat_type} {stat_category or 'stats'}")
            
            scraper = Fbref()
            
            if stat_type == "Equipos":
                try:
                    data = scraper.get_teams_season_stats(
                        stat=stat_category or 'stats',
                        league=league,
                        season=season
                    )
                except TypeError as te:
                    # Error común: "arg must be a list, tuple, 1-d array, or Series"
                    # Ocurre cuando FBRef devuelve tabla con estructura inesperada
                    logger.warning(f"[FBRef] Error de tipo en equipos: {te}. Intentando con jugadores...")
                    data = None
                
                # Fallback: si equipos viene vacío o falló, agregar datos de jugadores
                if data is None or (hasattr(data, 'empty') and data.empty):
                    logger.warning("[FBRef] Stats de equipos vacío, generando desde jugadores...")
                    data = _generate_team_stats_from_players(scraper, league, season)
            else:  # Jugadores
                try:
                    data = scraper.get_player_season_stats(
                        stat=stat_category or 'stats',
                        league=league,
                        season=season
                    )
                except TypeError as te:
                    logger.warning(f"[FBRef] Error de tipo en jugadores: {te}. Intentando stat='stats'...")
                    # Reintentar con stat por defecto
                    data = scraper.get_player_season_stats(
                        stat='stats',
                        league=league,
                        season=season
                    )
            
            # Validar y convertir resultado
            if data is not None:
                if isinstance(data, (list, tuple)):
                    # Si devuelve múltiples tablas, tomar la primera
                    data = data[0] if len(data) > 0 else pd.DataFrame()
                if not isinstance(data, pd.DataFrame):
                    try:
                        data = pd.DataFrame(data)
                    except Exception:
                        data = pd.DataFrame()
            
            logger.info(f"[FBRef] Datos obtenidos exitosamente")
            return data if data is not None else pd.DataFrame()
            
        except TypeError as te:
            # Capturar errores de tipo específicos de pandas
            error_msg = str(te)
            if "arg must be a list" in error_msg:
                logger.warning(f"[FBRef] Error de estructura de datos: {te}")
                # Intentar fallback a jugadores y agregar
                try:
                    scraper = Fbref()
                    data = _generate_team_stats_from_players(scraper, league, season)
                    if data is not None and not data.empty:
                        return data
                except Exception as fallback_err:
                    logger.warning(f"[FBRef] Fallback también falló: {fallback_err}")
            last_exception = te
            
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            
            # Errores que no vale la pena reintentar
            if any(err in error_msg for err in ['not found', 'invalid league', '404']):
                logger.error(f"[FBRef] Error no recuperable: {e}")
                raise
            
            if attempt < config["max_retries"] - 1:
                delay = config["base_delay"] * (2 ** attempt)
                logger.warning(f"[FBRef] Error: {e}. Esperando {delay}s antes de reintentar...")
                time.sleep(delay)
            else:
                logger.error(f"[FBRef] Fallo despues de {config['max_retries']} intentos: {e}")
    
    raise last_exception


def _generate_team_stats_from_players(scraper, league: str, season: str) -> pd.DataFrame:
    """
    Genera estadísticas de equipos agregando datos de jugadores.
    Usado como fallback cuando get_teams_season_stats devuelve vacío.
    """
    try:
        # Obtener datos de jugadores con manejo de errores
        try:
            players = scraper.get_player_season_stats(stat='stats', league=league, season=season)
        except TypeError as te:
            logger.warning(f"[FBRef] TypeError en get_player_season_stats: {te}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"[FBRef] Error obteniendo jugadores: {e}")
            return pd.DataFrame()
        
        if players is None:
            return pd.DataFrame()
        
        # Manejar caso donde devuelve lista/tupla
        if isinstance(players, (list, tuple)):
            players = players[0] if len(players) > 0 else pd.DataFrame()
        
        if not isinstance(players, pd.DataFrame):
            try:
                players = pd.DataFrame(players)
            except Exception:
                return pd.DataFrame()
        
        if players.empty:
            return pd.DataFrame()
        
        # Aplanar MultiIndex si existe
        if isinstance(players.columns, pd.MultiIndex):
            try:
                new_cols = []
                for col in players.columns:
                    if isinstance(col, tuple):
                        parts = [str(c) for c in col if str(c).strip() and 'Unnamed' not in str(c)]
                        new_cols.append('_'.join(parts) if parts else f'col_{len(new_cols)}')
                    else:
                        new_cols.append(str(col))
                players.columns = new_cols
            except Exception:
                players.columns = [f'col_{i}' for i in range(len(players.columns))]
        
        # Buscar columna de equipo
        squad_col = None
        for col in ['Squad', 'squad', 'Team', 'team']:
            if col in players.columns:
                squad_col = col
                break
        
        if not squad_col:
            # Buscar columna que contenga 'squad' o 'team' (case insensitive)
            for col in players.columns:
                if 'squad' in str(col).lower() or 'team' in str(col).lower():
                    squad_col = col
                    break
        
        if not squad_col:
            logger.warning("[FBRef] No se encontró columna de equipo")
            return pd.DataFrame()
        
        # Columnas numéricas para agregar
        numeric_cols = ['Gls', 'Ast', 'G+A', 'xG', 'xAG', 'MP', '90s', 'Starts', 'CrdY', 'CrdR']
        agg_dict = {}
        
        for col in numeric_cols:
            # Buscar columna exacta o que termine con el nombre
            matching_col = None
            if col in players.columns:
                matching_col = col
            else:
                for c in players.columns:
                    if str(c).endswith(col) or str(c).endswith(f'_{col}'):
                        matching_col = c
                        break
            
            if matching_col:
                try:
                    players[matching_col] = pd.to_numeric(players[matching_col], errors='coerce').fillna(0)
                    agg_dict[matching_col] = 'sum'
                except Exception:
                    pass
        
        # Si no hay columnas para agregar, retornar vacío
        if not agg_dict:
            logger.warning("[FBRef] No se encontraron columnas numéricas para agregar")
            return pd.DataFrame()
        
        # Agregar por equipo
        try:
            team_stats = players.groupby(squad_col).agg(agg_dict).reset_index()
            team_stats = team_stats.rename(columns={squad_col: 'Squad'})
        except Exception as e:
            logger.error(f"[FBRef] Error agregando datos: {e}")
            return pd.DataFrame()
        
        # Estimar goles en contra (aproximación basada en promedio de liga)
        gls_col = None
        for c in team_stats.columns:
            if 'Gls' in str(c) or 'gls' in str(c).lower():
                gls_col = c
                break
        
        if gls_col:
            avg_goals = team_stats[gls_col].mean()
            mp_col = None
            for c in team_stats.columns:
                if 'MP' in str(c) or 'mp' in str(c).lower():
                    mp_col = c
                    break
            
            if mp_col:
                avg_mp = team_stats[mp_col].mean()
                if avg_mp > 0:
                    team_stats['GA'] = (avg_goals * team_stats[mp_col] / avg_mp).round(0).astype(int)
            else:
                team_stats['GA'] = int(avg_goals)
            
            # Renombrar Gls a GF para consistencia
            team_stats['GF'] = team_stats[gls_col]
        
        logger.info(f"[FBRef] Generadas stats para {len(team_stats)} equipos desde jugadores")
        return team_stats
        
    except Exception as e:
        logger.error(f"[FBRef] Error general en _generate_team_stats_from_players: {e}")
        return pd.DataFrame()


def _extract_fotmob(league: str, season: str, stat_type: str) -> pd.DataFrame:
    """Extrae datos de FotMob."""
    scraper = FotMob()
    
    if stat_type == "Equipos":
        return scraper.get_teams_stats_season(league=league, season=season, stat='rating')
    elif stat_type == "Tablas":
        return scraper.get_season_tables(league=league, season=season, table='all')
    else:  # Jugadores
        return scraper.get_players_stats_season(league=league, season=season, stat='goals')


def fix_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas duplicadas agregando sufijos numéricos.
    Maneja MultiIndex (columnas jerárquicas) aplanándolas primero.
    
    Args:
        df: DataFrame con posibles columnas duplicadas
    
    Returns:
        DataFrame con columnas únicas
    """
    try:
        # Si el DataFrame está vacío o no tiene columnas válidas
        if df is None:
            return pd.DataFrame()
        
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        if df.empty or len(df.columns) == 0:
            return df
        
        # Manejar MultiIndex (común en tablas de FBRef)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # Aplanar MultiIndex a strings simples
                new_cols = []
                for col in df.columns:
                    if isinstance(col, tuple):
                        # Filtrar partes vacías y 'Unnamed'
                        parts = [str(c) for c in col if str(c).strip() != '' and 'Unnamed' not in str(c)]
                        new_cols.append('_'.join(parts) if parts else f'col_{len(new_cols)}')
                    else:
                        new_cols.append(str(col))
                df.columns = new_cols
            except Exception as e:
                logger.warning(f"Error aplanando MultiIndex: {e}. Usando índices numéricos.")
                df.columns = [f'col_{i}' for i in range(len(df.columns))]
        
        # Convertir columnas a lista de strings de forma segura
        try:
            col_list = [str(c) if c is not None else f'col_{i}' for i, c in enumerate(df.columns)]
        except Exception:
            col_list = [f'col_{i}' for i in range(len(df.columns))]
        
        # Verificar que tenemos una lista válida
        if not col_list:
            return df
        
        cols = pd.Series(col_list)
        
        # Encontrar y renombrar duplicados
        duplicated_mask = cols.duplicated()
        if duplicated_mask.any():
            for dup in cols[duplicated_mask].unique():
                # Obtener índices de las columnas duplicadas
                dup_indices = cols[cols == dup].index.tolist()
                # Renombrar con sufijos
                for i, idx in enumerate(dup_indices):
                    if i == 0:
                        continue  # Mantener el primer nombre sin cambio
                    cols.iloc[idx] = f"{dup}_{i+1}"
        
        df.columns = cols.tolist()
        return df
        
    except Exception as e:
        logger.warning(f"Error procesando columnas duplicadas: {e}. Retornando DataFrame original.")
        return df

