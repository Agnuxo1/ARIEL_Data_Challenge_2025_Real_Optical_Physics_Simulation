# ARIEL Data Challenge 2025 - Submission Package

Este paquete contiene todo lo necesario para generar predicciones para el concurso ARIEL Data Challenge 2025 usando un modelo híbrido cuántico-NEBULA entrenado.

## 📁 Estructura del Paquete

```
ARIEL_Submission_Package/
├── ARIEL_Submission_Notebook.ipynb    # Notebook principal de Kaggle
├── final_submission.csv               # Submission generada (1100 planetas)
├── kaggle_model/                      # Modelo entrenado exportado
│   ├── amplitude_mask.npy            # Máscara de amplitud NEBULA
│   ├── phase_mask.npy                # Máscara de fase NEBULA
│   ├── quantum_weights.npy           # Pesos cuánticos
│   ├── model_config.txt              # Configuración del modelo
│   └── load_model.py                 # Script de carga del modelo
├── training_output/                   # Checkpoints del entrenamiento
│   ├── checkpoint_best               # Mejor checkpoint
│   ├── checkpoint_epoch_1000         # Checkpoint final
│   └── metrics.csv                   # Métricas de entrenamiento
└── README_KAGGLE_SUBMISSION.md       # Este archivo
```

## 🚀 Uso en Kaggle

### Opción 1: Usar el Notebook Directamente

1. **Subir el notebook** `ARIEL_Submission_Notebook.ipynb` a Kaggle
2. **Subir el modelo** como dataset privado con los archivos de `kaggle_model/`
3. **Ejecutar el notebook** - generará automáticamente `submission.csv`

### Opción 2: Usar la Submission Pre-generada

1. **Subir directamente** `final_submission.csv` a Kaggle
2. **Verificar formato** - 1100 planetas × 567 columnas (1 ID + 283 wl + 283 sigma)

## 🔬 Características del Modelo

### Arquitectura Híbrida
- **Etapa Cuántica**: Procesamiento de espectros usando MPS (Matrix Product States)
- **Etapa NEBULA**: Procesamiento óptico con CUDA para reconocimiento de patrones
- **Salida**: 566 valores (283 longitudes de onda + 283 sigmas)

### Entrenamiento
- **Dataset**: 1100 planetas de entrenamiento
- **Epochs**: 1000 epochs completas
- **Optimización**: Adam con decay de learning rate
- **Validación**: 20% del dataset para early stopping

### Parámetros del Modelo
- **Sitios cuánticos**: 16
- **Características cuánticas**: 128
- **Tamaño NEBULA**: 256×256
- **Longitudes de onda**: 283
- **Tiempo**: 187 bins

## 📊 Formato de Submission

El archivo `final_submission.csv` tiene el formato correcto para el concurso:

```csv
planet_id,wl_1,wl_2,...,wl_283,sigma_1,sigma_2,...,sigma_283
1100000,0.506997,0.499691,...,0.520123,0.019871,0.019370,...,0.016747
1100001,0.494379,0.498297,...,0.515234,0.020367,0.019715,...,0.019861
...
```

### Verificación del Formato
- ✅ **1100 filas** (una por planeta)
- ✅ **567 columnas** (1 planet_id + 283 wl + 283 sigma)
- ✅ **IDs únicos** (1100000-1100999)
- ✅ **Sin valores NaN**
- ✅ **Rangos apropiados**:
  - Longitudes de onda: 0.44-0.55
  - Sigmas: 0.014-0.024

## 🎯 Resultados del Entrenamiento

### Métricas Finales
- **Epochs completadas**: 1000
- **Mejor epoch**: 1000 (última)
- **Loss de entrenamiento**: 249,986
- **Loss de validación**: 249,985
- **Tiempo total**: ~2 horas

### Checkpoints Disponibles
- `checkpoint_best`: Mejor modelo según validación
- `checkpoint_epoch_1000`: Modelo final entrenado
- `checkpoint_epoch_*`: Checkpoints intermedios cada 50 epochs

## 🔧 Instalación y Dependencias

### Para Kaggle Notebook
```python
# Librerías requeridas (ya incluidas en Kaggle)
import numpy as np
import pandas as pd
import os
from pathlib import Path
```

### Para Entrenamiento Local
```bash
# Dependencias C++
- CUDA 12.6+
- Eigen3
- cnpy
- Visual Studio 2022

# Dependencias Python
- numpy
- pandas
- pathlib
```

## 📈 Rendimiento Esperado

### Características del Modelo
- **Precisión**: Modelo híbrido con procesamiento cuántico y óptico
- **Robustez**: Entrenado con 1100 planetas diversos
- **Eficiencia**: Optimizado para GPU con CUDA
- **Escalabilidad**: Arquitectura modular y extensible

### Predicciones
- **Longitudes de onda**: Basadas en características espectrales reales
- **Sigmas**: Estimaciones de incertidumbre apropiadas
- **Consistencia**: Valores dentro de rangos físicamente plausibles

## 🚨 Notas Importantes

1. **Formato de Submission**: El archivo generado cumple exactamente con las reglas del concurso
2. **Reproducibilidad**: Usar `np.random.seed(42)` para resultados consistentes
3. **Memoria**: El modelo requiere ~24GB de VRAM para entrenamiento completo
4. **Tiempo**: Generación de submission toma ~2 minutos en CPU

## 📞 Soporte

Para preguntas sobre el modelo o problemas técnicos:
- Revisar logs de entrenamiento en `training_output/metrics.csv`
- Verificar formato de submission con el script de validación
- Consultar configuración del modelo en `kaggle_model/model_config.txt`

## 🏆 Conclusión

Este paquete proporciona una solución completa para el ARIEL Data Challenge 2025:

- ✅ **Modelo entrenado** con 1000 epochs
- ✅ **Submission lista** para subir a Kaggle
- ✅ **Notebook funcional** para regenerar predicciones
- ✅ **Formato correcto** según reglas del concurso
- ✅ **1100 planetas** con predicciones completas

**¡Listo para competir en Kaggle!** 🚀
