# StatsBomb Data Utilities
# Funciones para acceder y procesar datos de StatsBomb

from statsbombpy import sb
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600)
def get_available_competitions():
    """
    Obtiene todas las competiciones disponibles en StatsBomb Open Data.
    Returns:
        DataFrame con las competiciones disponibles
    """
    try:
        competitions = sb.competitions()
        return competitions
    except Exception as e:
        st.error(f"Error al obtener competiciones: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_matches_for_competition(competition_id: int, season_id: int):
    """
    Obtiene los partidos de una competición y temporada específica.
    Args:
        competition_id: ID de la competición
        season_id: ID de la temporada
    Returns:
        DataFrame con los partidos
    """
    try:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        return matches
    except Exception as e:
        st.error(f"Error al obtener partidos: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_events_for_match(match_id: int):
    """
    Obtiene todos los eventos de un partido.
    Args:
        match_id: ID del partido
    Returns:
        DataFrame con todos los eventos del partido
    """
    try:
        events = sb.events(match_id=match_id)
        return events
    except Exception as e:
        st.error(f"Error al obtener eventos: {str(e)}")
        return pd.DataFrame()


def get_shots(events: pd.DataFrame):
    """
    Filtra los eventos de tipo disparo (Shot).
    Args:
        events: DataFrame de eventos
    Returns:
        DataFrame con solo los tiros
    """
    if events.empty:
        return pd.DataFrame()
    
    shots = events[events['type'] == 'Shot'].copy()
    return shots


def get_passes(events: pd.DataFrame, player_name: str = None, team_name: str = None):
    """
    Filtra los eventos de tipo pase (Pass).
    Args:
        events: DataFrame de eventos
        player_name: Nombre del jugador (opcional)
        team_name: Nombre del equipo (opcional)
    Returns:
        DataFrame con los pases filtrados
    """
    if events.empty:
        return pd.DataFrame()
    
    passes = events[events['type'] == 'Pass'].copy()
    
    if player_name:
        passes = passes[passes['player'] == player_name]
    
    if team_name:
        passes = passes[passes['team'] == team_name]
    
    return passes


def get_players_from_events(events: pd.DataFrame):
    """
    Obtiene lista única de jugadores de los eventos.
    Args:
        events: DataFrame de eventos
    Returns:
        Lista de nombres de jugadores
    """
    if events.empty or 'player' not in events.columns:
        return []
    
    players = events['player'].dropna().unique().tolist()
    return sorted(players)


def get_teams_from_events(events: pd.DataFrame):
    """
    Obtiene lista de equipos del partido.
    Args:
        events: DataFrame de eventos
    Returns:
        Lista de nombres de equipos
    """
    if events.empty or 'team' not in events.columns:
        return []
    
    teams = events['team'].dropna().unique().tolist()
    return sorted(teams)
