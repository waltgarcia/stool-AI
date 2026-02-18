# 📊 Analytics & Monitoring Guide

Este documento explica cómo monitorear la aplicación para entender su funcionamiento, número de ingresos y detectar problemas.

## 🎯 Componentes del Sistema de Analítica

### 1. **Analytics Module** (`analytics.py`)
Recopila datos sobre:
- **Sesiones de usuarios** - Cuándo entran y salten del sistema
- **Clasificaciones** - Qué imágenes se procesan y resultados
- **Feedback del usuario** - Correcciones a predicciones
- **Errores** - Problemas y bugs
- **Health checks** - Estado del sistema

### 2. **Analytics Dashboard** (`analytics_dashboard.py`)
Panel visual para ver:
- 📈 Estadísticas de uso en tiempo real
- 🏥 Estado de salud de la aplicación
- 📊 Gráficas y análisis
- ❌ Registro de errores
- ⚙️ Información del sistema

### 3. **Health Check Script** (`health_check.py`)
Verifica automáticamente:
- ✅ Si el modelo está cargado
- ✅ Si la base de datos funciona
- ✅ Espacio en disco disponible
- ✅ Estado de directorios
- ✅ Tiempo de respuesta

### 4. **Status Page** (`status_page.py`)
Página pública que muestra:
- 🟢 Estado operacional
- 📊 Métricas en tiempo real
- 📡 API JSON para integraciones

## 🚀 Cómo Usar

### Ver Analytics Dashboard

```bash
streamlit run analytics_dashboard.py
```

Acceda a: `http://localhost:8501/analytics_dashboard.py`

**Pestañas disponibles:**
1. **Overview** - Resumen rápido de uso
2. **Health** - Estado del sistema
3. **Statistics** - Análisis detallado
4. **Errors** - Historial de errores
5. **System** - Información del sistema

### Ejecutar Health Check

**Una sola verificación:**
```bash
python health_check.py
```

**Monitoreo continuo (cada 5 minutos):**
```bash
python health_check.py --continuous
```

**Intervalo personalizado (en segundos):**
```bash
python health_check.py --continuous 300  # Cada 5 minutos
python health_check.py --continuous 60   # Cada 1 minuto
```

### Ver Página de Estado

```bash
streamlit run status_page.py
```

Accede a: `http://localhost:8501/status_page.py`

## 📊 Métricas Disponibles

### Estadísticas de Uso

| Métrica | Descripción |
|---------|-------------|
| **Sesiones** | Número de usuarios únicos que visitaron |
| **Clasificaciones** | Total de imágenes procesadas |
| **Feedback** | Correcciones del usuario |
| **Errores** | Problemas detectados |
| **Precisión** | % de clasificaciones correctas |

### Métricas de Rendimiento

| Métrica | Descripción |
|---------|-------------|
| **Tiempo de respuesta** | ms que tarda en clasificar |
| **Confianza** | % promedio de confianza del modelo |
| **CPU** | Uso de procesador |
| **Memoria** | Uso de RAM |
| **Disco** | Espacio disponible |

### Salud del Sistema

| Métrica | Descripción |
|---------|-------------|
| **Modelo cargado** | Si el modelo está disponible |
| **Base de datos** | Si la BD funciona correctamente |
| **Directorios** | Si existen todos los directorios necesarios |
| **Disco** | MB disponibles en disco |

## 📍 Base de Datos de Analíticas

Los datos se guardan en: `data/analytics/analytics.db`

**Tablas:**
- `sessions` - Sesiones de usuario
- `events` - Eventos (clasificación, feedback, etc)
- `classifications` - Detalles de cada clasificación
- `errors` - Registro de errores
- `health_checks` - Verificaciones de salud

## 📈 Entender los Gráficos

### Overview
- **Daily Sessions** - Tendencia de visitantes diarios
- **Classification Distribution** - Qué tipos se clasifican más

### Health
- **CPU/Memory/Disk** - Recursos del servidor
- **Response Time** - Velocidad de procesamiento

### Statistics
- **Accuracy** - Qué % de predicciones fueron correctas
- **Performance** - Velocidad promedio de procesamiento

### Errors
- **Error by Type** - Qué tipos de errores ocurren
- **Recent Errors** - Últimos errores registrados

## 🔔 Alertas Automáticas

El sistema alertan sobre:
- ⚠️ Disco lleno (>90%)
- ⚠️ Modelo no cargado
- ⚠️ Base de datos corrupta
- ⚠️ Tiempo de respuesta lento (>1000ms)

## 🔗 Integración con Servicios Externos

### Opción 1: Uptime Robot
Monitorea: `status_page.py`

```
URL: https://tu-dominio.com/status_page.py
Método: GET
Intervalo: 5 minutos
```

### Opción 2: Grafana
Conecta la base de datos SQLite:
```
Tipo: SQLite
Archivo: data/analytics/analytics.db
```

### Opción 3: Monitoreo Manual
```bash
# Cron job cada 5 minutos
*/5 * * * * cd /ruta/a/stool-AI && python health_check.py >> /var/log/stool-ai-health.log 2>&1

# Cron job diario para reporte
0 0 * * * cd /ruta/a/stool-AI && python health_check.py
```

## 📊 Ejemplos de Uso

### Ver cuántos usuarios visitaron hoy
```python
from analytics import analytics

today_sessions = analytics.get_sessions_count(days=1)
print(f"Usuarios hoy: {today_sessions}")
```

### Ver precisión del modelo
```python
accuracy = analytics.get_accuracy_stats(days=7)
print(f"Precisión: {accuracy['accuracy']:.1f}%")
print(f"Correctas: {accuracy['correct']}/{accuracy['total']}")
```

### Ver errores recientes
```python
errors = analytics.get_errors_by_type(days=7)
print(errors)
```

### Exportar datos
```
En Analytics Dashboard → System → "Export Analytics Data"
```

## 🐛 Troubleshooting

### Error: "No analytics data"
- Verifica que hay sesiones activas
- Asegúrate que la BD existe en `data/analytics/analytics.db`
- Ejecuta la app principal: `streamlit run streamlit_app.py`

### Health check falla
- Verifica que `model_weights.pth` existe
- Confirma que directorios de datos existen
- Revisa espacio en disco: `df -h`

### Dashboard lento
- Reduce rango de días analizados
- Limpia datos antiguos (>90 días)
- Verifica espacio en disco

## 📋 Checklist de Monitoreo

**Diariamente:**
- [ ] Ver dashboard Overview
- [ ] Revisar si hay errores nuevos
- [ ] Verificar salud del sistema

**Semanalmente:**
- [ ] Analizar tendencias de uso
- [ ] Revisar accuracy del modelo
- [ ] Exportar estadísticas

**Mensualmente:**
- [ ] Generar reporte de uso
- [ ] Revisar crecimiento de datos
- [ ] Ajustar alertas si es necesario
- [ ] Limpiar datos antiguos si es necesario

## 🎯 KPIs Importantes

Métrica | Objetivo | Actual
--------|----------|--------
**Sesiones/mes** | >100 | TBD
**Uptime** | >99% | TBD
**Precision** | >90% | TBD
**Resp. Time** | <500ms | TBD
**Errores/mes** | <5 | TBD

---

**Última actualización:** 2024
**Versión:** 1.0
