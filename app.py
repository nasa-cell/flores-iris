"""
Clasificador de flores Iris — servidor Flask
---------------------------------------------
Sirve la página (predice con TensorFlow.js, sin servidor) y expone
/api/corregir: ajusta el modelo con model.fit() y guarda los pesos en
modelo_web/pesos_modelo.json. Con GITHUB_TOKEN configurado, también los
sube a GitHub, para que sobrevivan a un reinicio de Render.
"""

import base64
import json
import os
import threading

import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__, static_folder="estaticos", static_url_path="/estaticos")

RUTA_PESOS = "modelo_web/pesos_modelo.json"
RUTA_H5 = "modelo_iris.h5"
ESPECIES = ["Setosa", "Versicolor", "Virginica"]

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "nasa-cell/flores-iris")
GITHUB_RAMA = os.environ.get("GITHUB_BRANCH", "main")

# Protege al modelo de que dos correcciones lleguen al mismo tiempo desde
# distintos visitantes.
candado_modelo = threading.Lock()


def construir_modelo():
    modelo = keras.Sequential(
        [
            layers.Input(shape=(4,)),
            layers.Dense(8, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ]
    )
    modelo.compile(optimizer=keras.optimizers.Adam(0.05), loss="sparse_categorical_crossentropy")
    return modelo


with open(RUTA_PESOS, encoding="utf-8") as archivo:
    estado_inicial = json.load(archivo)

# Se guarda una copia de los pesos originales (los que vinieron de
# entrenar_modelo.py) para poder restablecerlos si hace falta.
pesos_originales = estado_inicial["pesos"]
media = np.array(estado_inicial["media"], dtype="float32")
desviacion = np.array(estado_inicial["desviacion"], dtype="float32")

modelo = construir_modelo()
modelo.set_weights([np.array(capa, dtype="float32") for capa in estado_inicial["pesos"]])

# "Calentamos" el modelo con un dato descartable: la primera llamada a
# model.fit() en un proceso es lenta (más aún con poco CPU, como en
# Render), así que mejor pagarla acá que en la primera corrección real.
modelo.fit(np.zeros((1, 4), dtype="float32"), np.array([0]), epochs=1, verbose=0)
modelo.set_weights([np.array(capa, dtype="float32") for capa in estado_inicial["pesos"]])


def subir_pesos_a_github(mensaje_commit):
    """Sube modelo_web/pesos_modelo.json al repositorio de GitHub. Sin
    GITHUB_TOKEN configurado, no hace nada."""
    if not GITHUB_TOKEN:
        return False

    url_api = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{RUTA_PESOS}"
    cabeceras = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    try:
        with open(RUTA_PESOS, "rb") as archivo:
            contenido_codificado = base64.b64encode(archivo.read()).decode("ascii")

        # Hace falta el sha del archivo actual en GitHub para poder
        # reemplazarlo (así lo pide la API de GitHub).
        respuesta_actual = requests.get(
            url_api, headers=cabeceras, params={"ref": GITHUB_RAMA}, timeout=15
        )
        sha_actual = respuesta_actual.json().get("sha") if respuesta_actual.ok else None

        cuerpo = {"message": mensaje_commit, "content": contenido_codificado, "branch": GITHUB_RAMA}
        if sha_actual:
            cuerpo["sha"] = sha_actual

        respuesta = requests.put(url_api, headers=cabeceras, json=cuerpo, timeout=15)
        return respuesta.ok
    except requests.RequestException:
        return False


def guardar_pesos_actuales(mensaje_commit):
    pesos = [capa.tolist() for capa in modelo.get_weights()]
    with open(RUTA_PESOS, "w", encoding="utf-8") as archivo:
        json.dump(
            {"especies": ESPECIES, "media": media.tolist(), "desviacion": desviacion.tolist(), "pesos": pesos},
            archivo,
        )
    modelo.save(RUTA_H5)
    return subir_pesos_a_github(mensaje_commit)


@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory(".", "index.html")


@app.route("/datos.html")
def pagina_datos():
    return send_from_directory(".", "datos.html")


@app.route("/modelo_web/<path:nombre_archivo>")
def modelo_web(nombre_archivo):
    respuesta = send_from_directory("modelo_web", nombre_archivo)
    # Sin caché: si el modelo se corrigió, el navegador tiene que traer la
    # versión nueva sí o sí, no una copia vieja guardada localmente.
    respuesta.headers["Cache-Control"] = "no-store"
    return respuesta


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
        modelo.fit(entrada, etiqueta, epochs=15, verbose=0)
        guardado_en_github = guardar_pesos_actuales(
            f"Corrige el modelo: {valores} -> {especie_correcta}"
        )

    return jsonify({"ok": True, "guardado_en_github": guardado_en_github})


@app.route("/api/restablecer", methods=["POST"])
def restablecer():
    with candado_modelo:
        modelo.set_weights([np.array(capa, dtype="float32") for capa in pesos_originales])
        guardado_en_github = guardar_pesos_actuales("Restablece el modelo a los pesos originales")

    return jsonify({"ok": True, "guardado_en_github": guardado_en_github})


if __name__ == "__main__":
    # Render asigna el puerto por la variable PORT; localmente usa 5000.
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto, debug=False)
