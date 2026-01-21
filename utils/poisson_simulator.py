# Simulador de Temporada de Fútbol con Distribución de Poisson
# Proyecta resultados de temporadas futuras basándose en datos históricos

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TeamStrength:
    """Representa la fuerza ofensiva y defensiva de un equipo."""
    name: str
    attack: float  # λ de ataque (goles promedio marcados)
    defense: float  # λ de defensa (goles promedio recibidos)
    home_factor: float = 1.2  # Factor de ventaja local


class PoissonSimulator:
    """
    Simulador de temporada usando distribución de Poisson.
    
    El modelo asume que los goles siguen una distribución de Poisson donde:
    - λ_ataque = Goles marcados / Partidos jugados
    - λ_defensa = Goles recibidos / Partidos jugados
    
    Para simular A vs B:
    - λ_A = λ_ataque(A) × (λ_defensa(B) / media_liga) × factor_local
    - λ_B = λ_ataque(B) × (λ_defensa(A) / media_liga)
    """
    
    def __init__(self, home_advantage: float = 1.25):
        """
        Inicializa el simulador.
        
        Args:
            home_advantage: Factor de ventaja local (default 1.25 = 25% más)
        """
        self.home_advantage = home_advantage
        self.teams: Dict[str, TeamStrength] = {}
        self.league_avg_goals = 0.0
        
    def calculate_team_strengths(self, df: pd.DataFrame, 
                                  team_col: str = 'Squad',
                                  goals_for_col: str = 'Gls',
                                  goals_against_col: str = 'GA',
                                  matches_col: str = 'MP') -> Dict[str, TeamStrength]:
        """
        Calcula las fortalezas de ataque y defensa para cada equipo.
        
        Args:
            df: DataFrame con estadísticas de equipos
            team_col: Columna con nombre del equipo
            goals_for_col: Columna con goles a favor
            goals_against_col: Columna con goles en contra
            matches_col: Columna con partidos jugados
            
        Returns:
            Diccionario de TeamStrength por equipo
        """
        self.teams = {}
        
        # Buscar columnas alternativas si no existen las especificadas
        available_cols = df.columns.tolist()
        
        # Buscar columna de equipo
        if team_col not in available_cols:
            for alt in ['Team', 'team', 'Equipo', 'Squad']:
                if alt in available_cols:
                    team_col = alt
                    break
        
        # Buscar columna de goles a favor
        if goals_for_col not in available_cols:
            for alt in ['GF', 'Goals', 'Goles', 'G', 'Gls']:
                if alt in available_cols:
                    goals_for_col = alt
                    break
        
        # Buscar columna de goles en contra
        if goals_against_col not in available_cols:
            for alt in ['GC', 'GoalsAgainst', 'Goals Against', 'GA']:
                if alt in available_cols:
                    goals_against_col = alt
                    break
        
        # Buscar columna de partidos
        if matches_col not in available_cols:
            for alt in ['Matches', 'PJ', 'Games', 'P', 'MP']:
                if alt in available_cols:
                    matches_col = alt
                    break
        
        # Verificar que tenemos las columnas necesarias
        required = [team_col, goals_for_col, goals_against_col, matches_col]
        missing = [c for c in required if c not in available_cols]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}. Disponibles: {available_cols}")
        
        # Convertir columnas numéricas a float (pueden venir como string)
        df = df.copy()
        for col in [goals_for_col, goals_against_col, matches_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calcular promedios de liga
        total_goals = df[goals_for_col].sum()
        total_matches = df[matches_col].sum() / 2  # Cada partido cuenta 2 veces
        self.league_avg_goals = total_goals / total_matches if total_matches > 0 else 1.5
        
        # Calcular fortaleza por equipo
        for _, row in df.iterrows():
            team_name = str(row[team_col])
            matches = float(row[matches_col])
            
            if matches > 0:
                attack = float(row[goals_for_col]) / matches
                defense = float(row[goals_against_col]) / matches
            else:
                attack = self.league_avg_goals
                defense = self.league_avg_goals
            
            self.teams[team_name] = TeamStrength(
                name=team_name,
                attack=attack,
                defense=defense,
                home_factor=self.home_advantage
            )
        
        return self.teams
    
    def simulate_match(self, home_team: str, away_team: str) -> Tuple[int, int]:
        """
        Simula un partido individual usando Poisson.
        
        Args:
            home_team: Nombre del equipo local
            away_team: Nombre del equipo visitante
            
        Returns:
            Tupla (goles_local, goles_visitante)
        """
        home = self.teams.get(home_team)
        away = self.teams.get(away_team)
        
        if not home or not away:
            raise ValueError(f"Equipo no encontrado: {home_team if not home else away_team}")
        
        # Calcular λ para cada equipo
        # λ_home = ataque_local × (defensa_visitante / media) × factor_local
        # λ_away = ataque_visitante × (defensa_local / media)
        
        avg = self.league_avg_goals if self.league_avg_goals > 0 else 1.5
        
        lambda_home = home.attack * (away.defense / avg) * home.home_factor
        lambda_away = away.attack * (home.defense / avg)
        
        # Limitar λ para evitar valores extremos
        lambda_home = max(0.1, min(lambda_home, 5.0))
        lambda_away = max(0.1, min(lambda_away, 5.0))
        
        # Simular goles con Poisson
        goals_home = np.random.poisson(lambda_home)
        goals_away = np.random.poisson(lambda_away)
        
        return int(goals_home), int(goals_away)
    
    def simulate_season(self, num_simulations: int = 1000, 
                        progress_callback=None) -> pd.DataFrame:
        """
        Simula múltiples temporadas completas usando Monte Carlo.
        
        Args:
            num_simulations: Número de temporadas a simular
            progress_callback: Función opcional para reportar progreso
            
        Returns:
            DataFrame con estadísticas agregadas por equipo
        """
        team_names = list(self.teams.keys())
        n_teams = len(team_names)
        
        if n_teams < 2:
            raise ValueError("Se necesitan al menos 2 equipos para simular")
        
        # Matrices para almacenar resultados
        points_matrix = np.zeros((num_simulations, n_teams))
        wins_matrix = np.zeros((num_simulations, n_teams))
        draws_matrix = np.zeros((num_simulations, n_teams))
        losses_matrix = np.zeros((num_simulations, n_teams))
        goals_for_matrix = np.zeros((num_simulations, n_teams))
        goals_against_matrix = np.zeros((num_simulations, n_teams))
        positions_matrix = np.zeros((num_simulations, n_teams))
        
        team_idx = {name: i for i, name in enumerate(team_names)}
        
        for sim in range(num_simulations):
            if progress_callback and sim % 100 == 0:
                progress_callback(sim / num_simulations)
            
            # Reiniciar estadísticas para esta simulación
            points = np.zeros(n_teams)
            wins = np.zeros(n_teams)
            draws = np.zeros(n_teams)
            losses = np.zeros(n_teams)
            gf = np.zeros(n_teams)
            ga = np.zeros(n_teams)
            
            # Simular todos los partidos (ida y vuelta)
            for i, home in enumerate(team_names):
                for j, away in enumerate(team_names):
                    if i != j:
                        goals_h, goals_a = self.simulate_match(home, away)
                        
                        gf[i] += goals_h
                        ga[i] += goals_a
                        gf[j] += goals_a
                        ga[j] += goals_h
                        
                        if goals_h > goals_a:
                            points[i] += 3
                            wins[i] += 1
                            losses[j] += 1
                        elif goals_h < goals_a:
                            points[j] += 3
                            wins[j] += 1
                            losses[i] += 1
                        else:
                            points[i] += 1
                            points[j] += 1
                            draws[i] += 1
                            draws[j] += 1
            
            # Calcular posiciones
            # Ordenar por puntos, luego diferencia de goles, luego goles a favor
            diff = gf - ga
            sort_keys = points * 1000000 + diff * 1000 + gf
            positions = np.argsort(-sort_keys) + 1  # 1-indexed
            
            # Guardar resultados
            points_matrix[sim] = points
            wins_matrix[sim] = wins
            draws_matrix[sim] = draws
            losses_matrix[sim] = losses
            goals_for_matrix[sim] = gf
            goals_against_matrix[sim] = ga
            
            for i, pos in enumerate(np.argsort(-sort_keys)):
                positions_matrix[sim, pos] = i + 1
        
        # Calcular estadísticas agregadas
        results = []
        for i, team in enumerate(team_names):
            pos_counts = positions_matrix[:, i]
            
            results.append({
                'Equipo': team,
                'Pts (Prom)': round(points_matrix[:, i].mean(), 1),
                'Pts (Min)': int(points_matrix[:, i].min()),
                'Pts (Max)': int(points_matrix[:, i].max()),
                'Pos (Prom)': round(pos_counts.mean(), 1),
                'Pos (Mejor)': int(pos_counts.min()),
                'Pos (Peor)': int(pos_counts.max()),
                '% Campeón': round((pos_counts == 1).sum() / num_simulations * 100, 1),
                '% Top 4': round((pos_counts <= 4).sum() / num_simulations * 100, 1),
                '% Descenso': round((pos_counts >= n_teams - 2).sum() / num_simulations * 100, 1),
                'GF (Prom)': round(goals_for_matrix[:, i].mean(), 1),
                'GC (Prom)': round(goals_against_matrix[:, i].mean(), 1),
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Pos (Prom)')
        
        return results_df
    
    def get_position_probabilities(self, num_simulations: int = 1000) -> pd.DataFrame:
        """
        Calcula la probabilidad de cada equipo de terminar en cada posición.
        
        Args:
            num_simulations: Número de simulaciones
            
        Returns:
            DataFrame con probabilidades por posición
        """
        team_names = list(self.teams.keys())
        n_teams = len(team_names)
        
        # Matriz de conteo de posiciones
        position_counts = np.zeros((n_teams, n_teams))
        
        for _ in range(num_simulations):
            points = np.zeros(n_teams)
            gf = np.zeros(n_teams)
            ga = np.zeros(n_teams)
            
            # Simular temporada
            for i, home in enumerate(team_names):
                for j, away in enumerate(team_names):
                    if i != j:
                        goals_h, goals_a = self.simulate_match(home, away)
                        gf[i] += goals_h
                        ga[i] += goals_a
                        if goals_h > goals_a:
                            points[i] += 3
                        elif goals_h < goals_a:
                            points[j] += 3
                        else:
                            points[i] += 1
                            points[j] += 1
            
            # Ordenar y contar posiciones
            diff = gf - ga
            sort_keys = points * 1000000 + diff * 1000 + gf
            for pos, team_idx in enumerate(np.argsort(-sort_keys)):
                position_counts[team_idx, pos] += 1
        
        # Convertir a probabilidades
        probs = position_counts / num_simulations * 100
        
        # Crear DataFrame
        columns = [f'Pos {i+1}' for i in range(n_teams)]
        df = pd.DataFrame(probs, index=team_names, columns=columns)
        df = df.round(1)
        
        return df


def create_poisson_simulator(df: pd.DataFrame, **kwargs) -> PoissonSimulator:
    """
    Función de conveniencia para crear un simulador desde un DataFrame.
    
    Args:
        df: DataFrame con estadísticas de equipos
        **kwargs: Argumentos para calculate_team_strengths
        
    Returns:
        PoissonSimulator configurado
    """
    simulator = PoissonSimulator()
    simulator.calculate_team_strengths(df, **kwargs)
    return simulator
