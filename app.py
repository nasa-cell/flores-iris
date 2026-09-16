"""
Clasificador de flores Iris — servidor Flask
---------------------------------------------
Carga modelo_web/modelo_iris.h5 con Keras y predice del lado del servidor
(/api/predecir). /api/corregir ajusta ese mismo modelo con train_on_batch()
y lo vuelve a guardar en modelo_iris.h5. Con GITHUB_TOKEN configurado,
también sube el .h5 actualizado a GitHub, para que sobreviva a un reinicio
de Render.
"""

import base64
import json
import os
import threading

import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory
from tensorflow import keras

app = Flask(__name__, static_folder="estaticos", static_url_path="/estaticos")

RUTA_CONFIG = "modelo_web/configuracion.json"
RUTA_H5 = "modelo_web/modelo_iris.h5"
ESPECIES = ["Setosa", "Versicolor", "Virginica"]

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "nasa-cell/flores-iris")
GITHUB_RAMA = os.environ.get("GITHUB_BRANCH", "main")

# Protege al modelo de que dos correcciones lleguen al mismo tiempo desde
# distintos visitantes.
candado_modelo = threading.Lock()


modelo = None
pesos_originales = None
media = None
desviacion = None


def descargar_modelo_de_github():
    """Trae la última versión de modelo_iris.h5 desde GitHub y la deja en
    RUTA_H5, ANTES de cargarla. El disco de Render es efímero: cada reinicio
    o redeploy vuelve a partir del .h5 que está en el repositorio, así que
    si no se descarga la versión corregida, cualquier corrección hecha con
    /api/corregir se perdería en el próximo reinicio. Sin GITHUB_TOKEN, o si
    la descarga falla, se sigue con el .h5 que ya está en el disco (el del
    propio repositorio/checkout)."""
    if not GITHUB_TOKEN:
        return False

    url_api = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{RUTA_H5}"
    cabeceras = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    try:
        respuesta = requests.get(url_api, headers=cabeceras, params={"ref": GITHUB_RAMA}, timeout=(5, 15))
        if not respuesta.ok:
            return False
        contenido = base64.b64decode(respuesta.json()["content"])
        with open(RUTA_H5, "wb") as archivo:
            archivo.write(contenido)
        print("Modelo descargado desde GitHub (última corrección incluida).", flush=True)
        return True
    except (requests.RequestException, KeyError, ValueError) as error:
        print("descargar_modelo_de_github: fallo, se usa el .h5 del repositorio:", repr(error), flush=True)
        return False


def inicializar_modelo():
    """Carga modelo_iris.h5 y lo calienta con un dato descartable. Gunicorn
    la llama desde gunicorn.conf.py (hook post_fork) para que TensorFlow se
    inicialice DESPUÉS de que el proceso se bifurque en cada worker — si
    se inicializa antes (en el proceso padre), los hilos internos de
    TensorFlow quedan rotos en el worker y el entrenamiento se cuelga."""
    global modelo, pesos_originales, media, desviacion

    descargar_modelo_de_github()

    with open(RUTA_CONFIG, encoding="utf-8") as archivo:
        configuracion = json.load(archivo)

    media = np.array(configuracion["media"], dtype="float32")
    desviacion = np.array(configuracion["desviacion"], dtype="float32")

    modelo = keras.models.load_model(RUTA_H5)
    # Mismo optimizador que usaba /api/corregir antes: Adam con learning
    # rate alto, para que una corrección puntual mueva el modelo en pocos
    # pasos. El modelo.save() de Keras guarda el optimizador con el que se
    # entrenó originalmente (uno más conservador), así que se recompila acá.
    modelo.compile(optimizer=keras.optimizers.Adam(0.05), loss="sparse_categorical_crossentropy")

    pesos_originales = [capa.copy() for capa in modelo.get_weights()]
    modelo.train_on_batch(np.zeros((1, 4), dtype="float32"), np.array([0]))
    modelo.set_weights(pesos_originales)
    print("Modelo listo.", flush=True)


def subir_modelo_a_github(mensaje_commit):
    """Sube modelo_web/modelo_iris.h5 al repositorio de GitHub. Sin
    GITHUB_TOKEN configurado, no hace nada."""
    if not GITHUB_TOKEN:
        return False

    url_api = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{RUTA_H5}"
    cabeceras = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    try:
        with open(RUTA_H5, "rb") as archivo:
            contenido_codificado = base64.b64encode(archivo.read()).decode("ascii")

        # Hace falta el sha del archivo actual en GitHub para poder
        # reemplazarlo (así lo pide la API de GitHub).
        respuesta_actual = requests.get(
            url_api, headers=cabeceras, params={"ref": GITHUB_RAMA}, timeout=(5, 15)
        )
        sha_actual = respuesta_actual.json().get("sha") if respuesta_actual.ok else None

        cuerpo = {"message": mensaje_commit, "content": contenido_codificado, "branch": GITHUB_RAMA}
        if sha_actual:
            cuerpo["sha"] = sha_actual

        respuesta = requests.put(url_api, headers=cabeceras, json=cuerpo, timeout=(5, 15))
        return respuesta.ok
    except requests.RequestException as error:
        print("subir_modelo_a_github: fallo de red:", repr(error), flush=True)
        return False


def guardar_modelo_actual(mensaje_commit):
    modelo.save(RUTA_H5)
    return subir_modelo_a_github(mensaje_commit)


@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory("plantillas", "index.html")


@app.route("/datos.html")
def pagina_datos():
    return send_from_directory("plantillas", "datos.html")


@app.route("/api/predecir", methods=["POST"])
def predecir():
    datos = request.get_json(force=True, silent=True) or {}
    valores = datos.get("valores")

    if not isinstance(valores, list) or len(valores) != 4:
        return jsonify({"error": "Faltan las 4 medidas."}), 400
    try:
        entrada = (np.array(valores, dtype="float32") - media) / desviacion
    except (TypeError, ValueError):
        return jsonify({"error": "Las 4 medidas deben ser números."}), 400
    entrada = entrada.reshape(1, 4)

    with candado_modelo:
        probabilidades = modelo.predict(entrada, verbose=0)[0]

    indice = int(np.argmax(probabilidades))
    return jsonify(
        {
            "especie": ESPECIES[indice],
            "confianza": float(probabilidades[indice]),
            "probabilidades": {especie: float(p) for especie, p in zip(ESPECIES, probabilidades)},
        }
    )


@app.route("/api/corregir", methods=["POST"])
def corregir():
    datos = request.get_json(force=True, silent=True) or {}
    valores = datos.get("valores")
    especie_correcta = datos.get("especie")

    if not isinstance(valores, list) or len(valores) != 4 or especie_correcta not in ESPECIES:
        return jsonify({"error": "Faltan datos o la especie no es válida."}), 400

    indice = ESPECIES.index(especie_correcta)
    entrada = (np.array(valores, dtype="float32") - media) / desviacion
    entrada = entrada.reshape(1, 4)
    etiqueta = np.array([indice])

    with candado_modelo:
        # 15 pasos con train_on_batch() en vez de fit(epochs=15): mismo
        # efecto (15 pasadas de descenso de gradiente sobre este dato),
        # pero sin el envoltorio de fit() que se cuelga en Render.
        for _ in range(15):
            modelo.train_on_batch(entrada, etiqueta)
        guardado_en_github = guardar_modelo_actual(
            f"Corrige el modelo: {valores} -> {especie_correcta}"
        )

    return jsonify({"ok": True, "guardado_en_github": guardado_en_github})


@app.route("/api/restablecer", methods=["POST"])
def restablecer():
    with candado_modelo:
        modelo.set_weights([capa.copy() for capa in pesos_originales])
        guardado_en_github = guardar_modelo_actual("Restablece el modelo a los pesos originales")

    return jsonify({"ok": True, "guardado_en_github": guardado_en_github})


if __name__ == "__main__":
    # Corriendo con "python app.py" (sin gunicorn ni fork de por medio) se
    # inicializa acá directamente.
    inicializar_modelo()
    # Render asigna el puerto por la variable PORT; localmente usa 5000.
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto, debug=False)
