# Clasificador de flores Iris

Página web que predice la especie de una flor Iris (Setosa, Versicolor o Virginica) a partir de sus 4 medidas clásicas, usando una red neuronal entrenada en Python con Keras.

## Arquitectura

```
HTML → JavaScript → API Flask → Keras → modelo_iris.h5 → predicción → HTML
```

`modelo_web/modelo_iris.h5` es el modelo Keras que realmente se usa para predecir — no una copia ni una reconstrucción. Flask lo carga una sola vez al iniciar (`keras.models.load_model()`) y lo mantiene en memoria; todas las predicciones y correcciones se hacen contra ese mismo objeto.

## Cómo funciona

1. Escribes las 4 medidas de la flor (o usas un ejemplo / los datos de referencia) y tocas **Predecir**.
2. El navegador manda esas 4 medidas a `/api/predecir`. Flask las normaliza (misma media/desviación con las que se entrenó) y le pide la predicción al modelo cargado desde `modelo_iris.h5`. La predicción corre siempre en el servidor, no en el navegador.
3. Después de predecir, puedes decir si acertó o no. Si no acertó, eliges la especie correcta y esa corrección se manda a `/api/corregir`: Flask ajusta el modelo con unos pocos pasos de `train_on_batch()` y vuelve a guardarlo en `modelo_iris.h5` — así, la próxima predicción de **cualquier visitante** ya sale con el modelo corregido, sin que el navegador tenga que hacer nada más.
4. Un botón "↺ Restablecer modelo original" (`/api/restablecer`) deshace todas las correcciones y vuelve a los pesos que tenía el modelo al arrancar el servidor.

## Endpoints

| Ruta | Qué hace |
|---|---|
| `/api/predecir` (POST) | Recibe `{"valores": [largo_sepalo, ancho_sepalo, largo_petalo, ancho_petalo]}` y devuelve la especie, la confianza y las probabilidades de cada especie. |
| `/api/corregir` (POST) | Recibe las 4 medidas y la especie correcta. Ajusta `modelo_iris.h5` con `train_on_batch()`, lo guarda y, si hay `GITHUB_TOKEN`, lo sube a GitHub. |
| `/api/restablecer` (POST) | Vuelve el modelo a los pesos que tenía al iniciar el proceso, lo guarda y también lo sube a GitHub si corresponde. |

## Estructura del proyecto

```
entrenar_modelo.py           entrena el modelo en Python (Keras) y genera lo de modelo_web/
app.py                       servidor Flask: carga el modelo, predice y expone /api/corregir y /api/restablecer
gunicorn.conf.py             carga el modelo después del fork de cada worker (ver Notas)
requirements.txt             dependencias de Python
render.yaml                  configuración para desplegar en Render

plantillas/
  index.html                 página principal (formulario + predicción)
  datos.html                 18 flores reales de ejemplo, agrupadas por especie, con foto

estaticos/
  css/estilos.css            estilos
  js/script.js                lógica del navegador (formulario, llamadas a /api/predecir y /api/corregir)
  img/                        fotos de referencia de cada especie

modelo_web/
  modelo_iris.h5              modelo Keras real, usado para predecir y corregir
  configuracion.json           media y desviación para normalizar los datos, y la lista de especies
```

## Correrlo en tu computadora

Necesitas Python 3.11+.

```bash
pip install -r requirements.txt
python app.py
```

Abre `http://127.0.0.1:5000/`.

Si quieres volver a entrenar el modelo desde cero (por ejemplo, para ver otra precisión o cambiar la arquitectura), corre `python entrenar_modelo.py` — necesita además `scikit-learn` instalado (`pip install scikit-learn`), que no hace falta en el servidor porque solo se usa para entrenar, no para predecir.

## Desplegarlo en Render

Este repositorio ya incluye `render.yaml`:

1. Entra a [render.com](https://render.com) y conecta tu cuenta de GitHub.
2. **New +** → **Blueprint** → elige este repositorio.
3. Render instala las dependencias y arranca el servidor solo.

**Importante sobre memoria:** el servidor carga TensorFlow para predecir y para corregir el modelo, lo que pesa más que una página web normal. Si el plan gratuito de Render (512 MB) se queda sin memoria, hay que subir a un plan con más RAM.

## Cómo sobreviven las correcciones a un reinicio de Render

El disco de un Web Service de Render es efímero: **se pierde en cada reinicio o redeploy**, y los discos persistentes no existen en el plan free. Por eso las correcciones no dependen del disco, sino de GitHub:

- Con `GITHUB_TOKEN` configurado, cada corrección (`/api/corregir`) y cada restablecimiento (`/api/restablecer`) sube `modelo_iris.h5` al repositorio de GitHub.
- Al iniciar, antes de cargar el modelo, Flask intenta descargar la versión más reciente de `modelo_iris.h5` desde GitHub y la usa en vez de la que vino con el último deploy. Así, si Render reinicia el servicio, la corrección más reciente no se pierde.
- Sin `GITHUB_TOKEN`, este paso simplemente no hace nada y el servidor sigue con el `.h5` que trae el propio repositorio (comportamiento normal para correrlo en tu computadora).

## Notas

- La predicción y la corrección corren siempre del lado del servidor, contra `modelo_iris.h5` — ya no se usa TensorFlow.js en el navegador.
- `gunicorn.conf.py` carga el modelo en un hook `post_fork`, no al importar `app.py`: TensorFlow tiene que inicializarse después de que gunicorn bifurca cada worker, o el entrenamiento se cuelga.
- `render.yaml` usa `--workers 1`: el modelo vive una sola vez en memoria RAM, y `/api/corregir` lo actualiza ahí mismo. Con más de 1 worker, cada proceso tendría su propia copia del modelo (usando más RAM) y las correcciones hechas en un worker no se verían reflejadas en las predicciones que atienda otro — por eso hay que mantener siempre `--workers 1`.
- El modelo fue evaluado con datos que nunca vio durante el entrenamiento: acierta 29 de 30 (96.7%) en el conjunto de prueba automático, y 17 de 18 (94.4%) en las 18 flores de `datos.html`.
