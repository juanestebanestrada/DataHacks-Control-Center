# Procesador GPU para cálculos estadísticos acelerados

import numpy as np

class GPUProcessor:
    """
    Procesador para cálculos estadísticos acelerados con GPU.
    Usa CuPy si está disponible, con fallback a NumPy.
    """
    
    def __init__(self):
        """Inicializa el procesador y detecta disponibilidad de GPU."""
        self.gpu_available = False
        self.xp = np  # Por defecto usa NumPy
        
        try:
            import cupy as cp
            # Verificar que hay una GPU disponible
            cp.cuda.Device(0).compute_capability
            self.xp = cp
            self.gpu_available = True
        except Exception:
            self.gpu_available = False
    
    def to_gpu(self, data):
        """Transfiere datos a GPU si está disponible."""
        if self.gpu_available:
            import cupy as cp
            if isinstance(data, np.ndarray):
                return cp.asarray(data)
            return data
        return data
    
    def to_cpu(self, data):
        """Transfiere datos de GPU a CPU."""
        if self.gpu_available:
            import cupy as cp
            if isinstance(data, cp.ndarray):
                return cp.asnumpy(data)
        return data
    
    def compute_statistics(self, data: np.ndarray) -> dict:
        """
        Calcula estadísticas descriptivas usando GPU si está disponible.
        
        Args:
            data: Array de numpy con los datos
        
        Returns:
            Diccionario con estadísticas calculadas
        """
        # Limpiar NaN
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return {'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0}
        
        try:
            # Intentar usar GPU
            gpu_data = self.to_gpu(clean_data)
            
            # Calcular estadísticas
            stats = {
                'mean': float(self.to_cpu(self.xp.mean(gpu_data))),
                'std': float(self.to_cpu(self.xp.std(gpu_data))),
                'median': float(self.to_cpu(self.xp.median(gpu_data))),
                'min': float(self.to_cpu(self.xp.min(gpu_data))),
                'max': float(self.to_cpu(self.xp.max(gpu_data)))
            }
        except Exception:
            # Fallback a NumPy si GPU falla (ej: CUDA Toolkit no instalado)
            self.gpu_available = False
            self.xp = np
            stats = {
                'mean': float(np.mean(clean_data)),
                'std': float(np.std(clean_data)),
                'median': float(np.median(clean_data)),
                'min': float(np.min(clean_data)),
                'max': float(np.max(clean_data))
            }
        
        return stats
    
    def normalize_for_radar(self, df, columns: list):
        """
        Normaliza columnas para gráfico radar (0-1 range) usando GPU.
        
        Args:
            df: DataFrame pandas
            columns: Lista de columnas a normalizar
        
        Returns:
            DataFrame con columnas normalizadas
        """
        import pandas as pd
        
        result = df.copy()
        
        for col in columns:
            if col in df.columns:
                try:
                    data = df[col].values.astype(float)
                    # Manejar NaN
                    mask = ~np.isnan(data)
                    
                    if mask.sum() > 0:
                        clean_data = data[mask]
                        
                        # Intentar GPU, fallback a NumPy
                        try:
                            gpu_data = self.to_gpu(clean_data)
                            min_val = float(self.to_cpu(self.xp.min(gpu_data)))
                            max_val = float(self.to_cpu(self.xp.max(gpu_data)))
                        except Exception:
                            self.gpu_available = False
                            self.xp = np
                            min_val = float(np.min(clean_data))
                            max_val = float(np.max(clean_data))
                        
                        if max_val - min_val > 0:
                            normalized = (data - min_val) / (max_val - min_val)
                        else:
                            normalized = np.zeros_like(data)
                        
                        result[col] = normalized
                except Exception:
                    # Si falla la conversión, mantener original
                    pass
        
        return result
    
    def compute_correlation_matrix(self, df, columns: list) -> np.ndarray:
        """
        Calcula matriz de correlación usando GPU.
        
        Args:
            df: DataFrame pandas
            columns: Lista de columnas para correlación
        
        Returns:
            Matriz de correlación como numpy array
        """
        import pandas as pd
        
        # Extraer datos numéricos
        data = df[columns].values.astype(float)
        
        # Reemplazar NaN con media de columna
        col_means = np.nanmean(data, axis=0)
        for i in range(data.shape[1]):
            mask = np.isnan(data[:, i])
            data[mask, i] = col_means[i]
        
        # Transferir a GPU
        gpu_data = self.to_gpu(data)
        
        # Calcular correlación
        if self.gpu_available:
            import cupy as cp
            corr = cp.corrcoef(gpu_data.T)
            return self.to_cpu(corr)
        else:
            return np.corrcoef(data.T)
    
    def compute_percentiles(self, data: np.ndarray, percentiles: list = [25, 50, 75]) -> dict:
        """
        Calcula percentiles usando GPU.
        
        Args:
            data: Array de datos
            percentiles: Lista de percentiles a calcular
        
        Returns:
            Diccionario con percentiles calculados
        """
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return {f"p{p}": 0 for p in percentiles}
        
        gpu_data = self.to_gpu(clean_data)
        
        result = {}
        for p in percentiles:
            result[f"p{p}"] = float(self.to_cpu(self.xp.percentile(gpu_data, p)))
        
        return result
    
    def batch_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalización por lotes (batch normalization) usando GPU.
        
        Args:
            data: Array 2D de datos (samples x features)
        
        Returns:
            Array normalizado
        """
        gpu_data = self.to_gpu(data.astype(float))
        
        # Calcular media y std por feature
        mean = self.xp.mean(gpu_data, axis=0, keepdims=True)
        std = self.xp.std(gpu_data, axis=0, keepdims=True)
        
        # Evitar división por cero
        std = self.xp.where(std == 0, 1, std)
        
        normalized = (gpu_data - mean) / std
        
        return self.to_cpu(normalized)
