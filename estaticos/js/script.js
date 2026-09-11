// Clasificador de flores Iris — lógica del navegador
//
// El modelo se entrenó en Python con Keras (ver entrenar_modelo.py) y se
// guardó como modelo_iris.h5. Acá reconstruimos esa misma red con
// TensorFlow.js (misma cantidad de capas y neuronas) y le cargamos los
// pesos ya entrenados desde modelo_web/pesos_modelo.json, para poder
// predecir directamente en el navegador, sin servidor.

const formulario = document.getElementById("formulario-medidas");
const botonPredecir = document.getElementById("boton-predecir");
const estadoModelo = document.getElementById("estado-modelo");
const resultado = document.getElementById("resultado");
const resultadoEspecie = document.getElementById("resultado-especie");
const resultadoConfianza = document.getElementById("resultado-confianza");
const mensajeError = document.getElementById("mensaje-error");

const campoLargoSepalo = document.getElementById("largo-sepalo");
const campoAnchoSepalo = document.getElementById("ancho-sepalo");
const campoLargoPetalo = document.getElementById("largo-petalo");
const campoAnchoPetalo = document.getElementById("ancho-petalo");

let modelo = null;
let media = null;
let desviacion = null;
let especies = null;

function mostrarError(texto) {
  mensajeError.textContent = texto;
  mensajeError.hidden = false;
}

function ocultarError() {
  mensajeError.hidden = true;
}

// Reconstruye la misma arquitectura que se entrenó en Python (entrenar_modelo.py:
// Dense(8, relu) -> Dense(8, relu) -> Dense(3, softmax)) y le carga los pesos
// ya entrenados en vez de empezar desde cero.
async function cargarModelo() {
  const respuesta = await fetch("modelo_web/pesos_modelo.json");
  if (!respuesta.ok) {
    throw new Error("No se pudo cargar el modelo entrenado.");
  }
  const datos = await respuesta.json();

  especies = datos.especies;
  media = datos.media;
  desviacion = datos.desviacion;

  modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 8, activation: "relu", inputShape: [4] }));
  modelo.add(tf.layers.dense({ units: 8, activation: "relu" }));
  modelo.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  // El orden de los pesos coincide con lo que devuelve model.get_weights()
  // en Keras: [pesos_capa1, sesgo_capa1, pesos_capa2, sesgo_capa2, ...].
  const pesosComoTensores = datos.pesos.map((capa) => tf.tensor(capa));
  modelo.setWeights(pesosComoTensores);
}

// Aplica la misma normalización (media 0, desviación 1) que se usó al
// entrenar, para que el modelo reciba los datos en la misma escala.
function normalizar(valores) {
  return valores.map((valor, i) => (valor - media[i]) / desviacion[i]);
}

async function predecir(valores) {
  const entrada = tf.tensor2d([normalizar(valores)]);
  const salida = modelo.predict(entrada);
  const probabilidades = await salida.data();
  entrada.dispose();
  salida.dispose();

  let indiceMax = 0;
  for (let i = 1; i < probabilidades.length; i++) {
    if (probabilidades[i] > probabilidades[indiceMax]) indiceMax = i;
  }

  return { especie: especies[indiceMax], confianza: probabilidades[indiceMax] };
}

function leerMedidas() {
  return [
    parseFloat(campoLargoSepalo.value),
    parseFloat(campoAnchoSepalo.value),
    parseFloat(campoLargoPetalo.value),
    parseFloat(campoAnchoPetalo.value),
  ];
}

async function predecirYMostrar(valores) {
  ocultarError();
  resultado.hidden = true;

  if (valores.some((v) => Number.isNaN(v))) {
    mostrarError("Completa las 4 medidas con números.");
    return;
  }

  try {
    const { especie, confianza } = await predecir(valores);
    resultadoEspecie.textContent = especie;
    resultadoEspecie.className = "resultado-especie especie-" + especie.toLowerCase();
    resultadoConfianza.textContent = `Confianza: ${(confianza * 100).toFixed(1)}%`;
    resultado.hidden = false;
  } catch (error) {
    mostrarError(error.message || "Ocurrió un error al predecir.");
  }
}

formulario.addEventListener("submit", (evento) => {
  evento.preventDefault();
  predecirYMostrar(leerMedidas());
});

document.querySelectorAll(".boton-ejemplo").forEach((boton) => {
  boton.addEventListener("click", () => {
    const valores = boton.dataset.valores.split(",").map(Number);
    campoLargoSepalo.value = valores[0];
    campoAnchoSepalo.value = valores[1];
    campoLargoPetalo.value = valores[2];
    campoAnchoPetalo.value = valores[3];
    predecirYMostrar(valores);
  });
});

cargarModelo()
  .then(() => {
    estadoModelo.textContent = "Modelo listo — escribe las medidas o prueba un ejemplo.";
    botonPredecir.disabled = false;
  })
  .catch((error) => {
    estadoModelo.textContent = "No se pudo cargar el modelo.";
    mostrarError(error.message || "No se pudo cargar el modelo.");
  });
