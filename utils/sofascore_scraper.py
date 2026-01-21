# Scraper para SofaScore usando API directa (sin navegador)
# NO usa ninguna función de LanusStats que abra navegador

import time
import requests
import pandas as pd

# IDs de ligas predefinidos para SofaScore
# Esto evita llamar a get_possible_leagues_for_page que abre el navegador
SOFASCORE_LEAGUES = {
    "Argentina Liga Profesional": {"id": 155, "seasons": {"2024": 57478, "2023": 48133, "2022": 40711}},
    "Premier League": {"id": 17, "seasons": {"2024": 61627, "2023": 52186, "2022": 41886}},
    "LaLiga": {"id": 8, "seasons": {"2024": 61643, "2023": 52376, "2022": 42409}},
    "Serie A": {"id": 23, "seasons": {"2024": 61641, "2023": 52530, "2022": 42415}},
    "Bundesliga": {"id": 35, "seasons": {"2024": 61639, "2023": 52608, "2022": 42268}},
    "Ligue 1": {"id": 34, "seasons": {"2024": 61632, "2023": 52571, "2022": 42273}},
    "Champions League": {"id": 7, "seasons": {"2024": 61644, "2023": 52162, "2022": 41897}},
    "Copa Libertadores": {"id": 384, "seasons": {"2024": 57317, "2023": 47954, "2022": 40370}},
    "MLS": {"id": 242, "seasons": {"2024": 57084, "2023": 47647, "2022": 40361}},
    "Brasileirao": {"id": 325, "seasons": {"2024": 58766, "2023": 48982, "2022": 40557}},
}

# Campos de estadísticas
LEAGUE_STATS_FIELDS = [
    'goals', 'yellowCards', 'redCards', 'groundDuelsWon',
    'assists', 'accuratePassesPercentage', 'minutesPlayed',
    'appearances', 'started', 'saves', 'cleanSheets',
    'interceptions', 'clearances', 'totalShots', 'shotsOnTarget',
    'expectedGoals', 'keyPasses', 'tackles'
]


class SofaScoreScraper:
    """
    Scraper de SofaScore usando API REST directa.
    NO usa navegador - todo es via HTTP requests.
    """
    
    def __init__(self):
        self.base_url = 'https://api.sofascore.com'
        self.session = requests.Session()
        self._setup_session()
        
    def _setup_session(self):
        """Configura la sesión con headers realistas."""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.sofascore.com/',
            'Origin': 'https://www.sofascore.com',
        })
    
    def _make_request(self, path: str, max_retries: int = 3) -> dict:
        """Realiza una solicitud a la API."""
        url = f"{self.base_url}/{path}"
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code in [403, 429]:
                    print(f"[SOFASCORE] Bloqueado/Rate limited, esperando...")
                    time.sleep(3)
                    self._rotate_user_agent()
                else:
                    print(f"[SOFASCORE] Status {response.status_code}")
                    
            except Exception as e:
                print(f"[SOFASCORE] Error: {e}")
                time.sleep(2)
        
        raise Exception(f"No se pudo conectar a SofaScore después de {max_retries} intentos")
    
    def _rotate_user_agent(self):
        """Rota el User-Agent."""
        import random
        uas = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15',
        ]
        self.session.headers['User-Agent'] = random.choice(uas)
    
    def get_available_leagues(self) -> list:
        """Retorna las ligas disponibles."""
        return list(SOFASCORE_LEAGUES.keys())
    
    def get_seasons_for_league(self, league: str) -> list:
        """Retorna las temporadas disponibles para una liga."""
        if league in SOFASCORE_LEAGUES:
            return list(SOFASCORE_LEAGUES[league]["seasons"].keys())
        return []
    
    def scrape_league_stats(
        self,
        league: str,
        season: str,
        accumulation: str = 'total'
    ) -> pd.DataFrame:
        """
        Extrae estadísticas de jugadores de una liga.
        """
        # Buscar IDs en diccionario predefinido
        if league not in SOFASCORE_LEAGUES:
            # Buscar coincidencia parcial
            for key in SOFASCORE_LEAGUES:
                if league.lower() in key.lower() or key.lower() in league.lower():
                    league = key
                    break
            else:
                raise ValueError(f"Liga '{league}' no encontrada. Disponibles: {list(SOFASCORE_LEAGUES.keys())}")
        
        league_data = SOFASCORE_LEAGUES[league]
        league_id = league_data["id"]
        
        if season not in league_data["seasons"]:
            available = list(league_data["seasons"].keys())
            raise ValueError(f"Temporada '{season}' no disponible. Disponibles: {available}")
        
        season_id = league_data["seasons"][season]
        
        # Construir campos
        fields_str = "%2C".join(LEAGUE_STATS_FIELDS)
        
        all_data = []
        offset = 0
        
        print(f"[SOFASCORE] Extrayendo: {league} ({season})")
        
        for page in range(20):
            url = (
                f'api/v1/unique-tournament/{league_id}/season/{season_id}/statistics'
                f'?limit=100&order=-rating&offset={offset}'
                f'&accumulation={accumulation}'
                f'&fields={fields_str}'
                f'&filters=position.in.G~D~M~F'
            )
            
            try:
                data = self._make_request(url)
                
                if 'results' not in data or not data['results']:
                    break
                
                for item in data['results']:
                    row = {
                        'player': item.get('player', {}).get('name', ''),
                        'team': item.get('team', {}).get('name', ''),
                        'position': item.get('player', {}).get('position', ''),
                        'rating': item.get('rating', 0),
                    }
                    # Agregar estadísticas
                    for field in LEAGUE_STATS_FIELDS:
                        row[field] = item.get(field, 0)
                    
                    all_data.append(row)
                
                print(f"[SOFASCORE] Página {page + 1}: {len(data['results'])} jugadores")
                
                if data.get('page', 1) >= data.get('pages', 1):
                    break
                
                offset += 100
                time.sleep(0.3)
                
            except Exception as e:
                print(f"[SOFASCORE] Error en página {page + 1}: {e}")
                break
        
        df = pd.DataFrame(all_data)
        print(f"[SOFASCORE] Total: {len(df)} jugadores extraídos")
        return df


def extract_sofascore_data(league: str, season: str) -> pd.DataFrame:
    """Función de conveniencia."""
    scraper = SofaScoreScraper()
    return scraper.scrape_league_stats(league, season)
