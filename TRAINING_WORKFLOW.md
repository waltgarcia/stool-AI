# 🤖 Training Data Workflow

Este documento explica cómo se recopilan, revisan y utilizan las imágenes de los usuarios para mejorar continuamente el modelo.

## 📊 Flujo Completo

```
Usuario sube imagen
        ↓
Modelo predice tipo
        ↓
Usuario confirma/corrige predicción
        ↓
Imagen se guarda en data/user_submissions/
        ↓
Admin revisa en admin_dashboard.py
        ↓
Clasifica manualmente si es necesario
        ↓
Mueve a data/bristol_stool_dataset/type_X/
        ↓
Ejecuta: python retrain_model.py
        ↓
Nuevo modelo entrena con todos los datos
        ↓
Model weights se actualizan automáticamente
```

## 🚀 Paso a Paso

### 1. Usuario carga imagen en streamlit_app.py

```python
# El usuario:
# - Sube una imagen
# - El modelo predice
# - Opcionalmente confirma/corrige la predicción
# - Hace click en "Save & Help Train Model"
```

**Qué sucede internamente:**
- La imagen se guarda en `data/user_submissions/`
- Se crea un registro en `data/user_submissions/submissions.csv` con:
  - timestamp
  - nombre del archivo
  - predicción del modelo
  - clasificación correcta (si el usuario la proporcionó)
  - feedback del usuario

### 2. Admin revisa los envíos

```bash
# En una terminal separada, ejecutar:
streamlit run admin_dashboard.py
```

**En el dashboard:**
- Ve la tab "Review Pending"
- Para cada imagen sin clasificar:
  - Ve la predicción del modelo
  - Selecciona la clasificación correcta
  - Opcionalmente lee el feedback del usuario
- Una vez revisadas todas, en la tab "Training Data":
  - Click en "Move all reviewed images to training dataset"
  - Las imágenes se mueven automáticamente a `data/bristol_stool_dataset/type_X/`

### 3. Reentrenar el modelo

```bash
python retrain_model.py
```

**El script:**
- Carga todos los datos de `data/bristol_stool_dataset/`
- Divide automáticamente en train/val/test (70/15/15)
- Entrena durante 30 épocas
- Guarda el mejor modelo en `model_weights.pth`
- Automáticamente la app cargará el nuevo modelo

## 📁 Estructura de Directorios

```
stool-AI/
├── streamlit_app.py              # App principal
├── admin_dashboard.py            # Panel de administración
├── retrain_model.py              # Script de reentrenamiento
│
├── data/
│   ├── user_submissions/         # Imágenes recibidas de usuarios
│   │   ├── submissions.csv       # Registro de todas las imágenes
│   │   ├── hash_timestamp_1.png
│   │   ├── hash_timestamp_2.png
│   │   └── ...
│   │
│   └── bristol_stool_dataset/    # Dataset para entrenar
│       ├── type_1/
│       │   ├── image_1.png
│       │   └── ...
│       ├── type_2/
│       │   └── ...
│       └── type_7/
│
└── model_weights.pth             # Pesos del modelo actual
```

## 🔐 Seguridad Admin Dashboard

El `admin_dashboard.py` está protegido con contraseña.

**Para usar localmente:**
1. La contraseña por defecto es `admin123` (CAMBIAR EN PRODUCCIÓN)
2. Para cambiarla, edita el archivo o usa variables de entorno

**Para usar en Streamlit Cloud:**
1. Crea un archivo `.streamlit/secrets.toml`:
```toml
admin_password = "tu_contraseña_segura"
```
2. En Streamlit Cloud, en Settings → Secrets, pega el contenido del archivo

## 📊 Monitoreo del Progreso

En el admin dashboard, puedes ver:
- Número total de envíos
- Envíos pendientes de revisión
- Imágenes ya en el dataset de entrenamiento
- Distribución de imágenes por tipo (gráfica)

## 💡 Mejores Prácticas

1. **Revisar regularmente**: Hazlo cada semana o cuando tengas ~20 imágenes nuevas
2. **Entrenar después de revisar**: Después de mover imágenes, ejecuta `retrain_model.py`
3. **Validar resultados**: Prueba la app después de entrenar para verificar mejora
4. **Mantener datos limpios**: Elimina imágenes mala calidad o confusas

## 🐛 Troubleshooting

### Error: "Dataset is empty"
- Verifica que hay imágenes en `data/bristol_stool_dataset/type_*/`
- Recuerda mover las imágenes desde user_submissions en el admin dashboard

### Error: "No images found"
- Asegúrate de que las imágenes están en las carpetas correctas
- Los nombres deben ser: type_1/, type_2/, ... type_7/

### Modelo no mejora
- Verifica que tienes suficientes imágenes (mínimo 5-10 por tipo)
- Asegúrate de que las clasificaciones son correctas
- Aumenta el número de épocas en `retrain_model.py`

## 🎯 Objetivos

- **Corto plazo**: Recopilar al menos 10-20 imágenes por tipo
- **Mediano plazo**: 50-100 imágenes por tipo para notable mejora
- **Largo plazo**: 200+ imágenes por tipo para máxima precisión
