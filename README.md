# Clasificador de flores Iris

Página web que predice la especie de una flor Iris (Setosa, Versicolor o Virginica) a partir de sus 4 medidas clásicas, usando una red neuronal entrenada en Python con Keras.

## Cómo funciona

1. Escribes las 4 medidas de la flor (o usas un ejemplo / los datos de referencia) y tocas **Predecir**.
2. La predicción corre en tu propio navegador con **TensorFlow.js**: se reconstruye la misma red que se entrenó en Python y se le cargan sus pesos ya entrenados (`modelo_web/pesos_modelo.json`). No hace falta esperar al servidor para predecir.
3. Después de predecir, puedes decir si acertó o no. Si no acertó, eliges la especie correcta y esa corrección se manda al servidor (`/api/corregir`), que ajusta el modelo y guarda los pesos nuevos — así, la próxima vez que **cualquier visitante** cargue la página, ya recibe el modelo corregido.
4. Un botón "↺ Restablecer modelo original" deshace todas las correcciones y vuelve a los pesos con los que se entrenó originalmente.

## Estructura del proyecto

```
entrenar_modelo.py           entrena el modelo en Python (Keras) y genera lo de modelo_web/
app.py                       servidor Flask: sirve la página y el endpoint de corrección
gunicorn.conf.py             carga el modelo después del fork de cada worker (ver Notas)
requirements.txt             dependencias de Python
render.yaml                  configuración para desplegar en Render

plantillas/
  index.html                 página principal (formulario + predicción)
  datos.html                 18 flores reales de ejemplo, agrupadas por especie, con foto

estaticos/
  css/estilos.css            estilos
  js/script.js                lógica del navegador (TensorFlow.js, formulario, corrección)
  img/                        fotos de referencia de cada especie

modelo_web/
  modelo_iris.h5              modelo entrenado (formato Keras)
  pesos_modelo.json           pesos del modelo + normalización, para usar en el navegador
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

**Importante sobre memoria:** el servidor carga TensorFlow para poder corregir el modelo (`/api/corregir`), lo que pesa más que una página web normal. Si el plan gratuito de Render (512 MB) se queda sin memoria, hay que subir a un plan con más RAM.

## Notas

- La predicción normal (sin corregir) no necesita servidor: corre en el navegador con TensorFlow.js.
- La corrección del modelo sí necesita servidor: ajusta el modelo y, con `GITHUB_TOKEN` configurado, sube el archivo corregido a este mismo repositorio — así sobrevive aunque Render reinicie el servicio.
- `gunicorn.conf.py` carga el modelo en un hook `post_fork`, no al importar `app.py`: TensorFlow tiene que inicializarse después de que gunicorn bifurca cada worker, o el entrenamiento se cuelga.
- El modelo fue evaluado con datos que nunca vio durante el entrenamiento: acierta 29 de 30 (96.7%) en el conjunto de prueba automático, y 17 de 18 (94.4%) en las 18 flores de `datos.html`.
