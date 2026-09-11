// Clasificador de flores Iris — lógica del navegador
//
// El modelo se entrenó en Python con Keras (ver entrenar_modelo.py) y se
// guardó como modelo_iris.h5. Acá reconstruimos esa misma red con
// TensorFlow.js (misma cantidad de capas y neuronas) y le cargamos los
// pesos ya entrenados desde modelo_web/pesos_modelo.json, para poder
// predecir directamente en el navegador, sin pasar por el servidor.
//
// La única parte que sí habla con el servidor (app.py) es la corrección:
// cuando alguien marca una predicción como equivocada, se manda a
// /api/corregir, que ajusta el modelo del lado del servidor y guarda el
// resultado en modelo_web/pesos_modelo.json — así, la próxima vez que
// cualquier visitante cargue la página, ya recibe el modelo corregido.

const formulario = document.getElementById("formulario-medidas");
const botonPredecir = document.getElementById("boton-predecir");
const estadoModelo = document.getElementById("estado-modelo");
const resultado = document.getElementById("resultado");
const resultadoEspecie = document.getElementById("resultado-especie");
const resultadoConfianza = document.getElementById("resultado-confianza");
const mensajeError = document.getElementById("mensaje-error");

const feedback = document.getElementById("feedback");
const botonAcerto = document.getElementById("boton-acerto");
const botonFallo = document.getElementById("boton-fallo");
const feedbackCorreccion = document.getElementById("feedback-correccion");
const feedbackMensaje = document.getElementById("feedback-mensaje");
const botonRestablecer = document.getElementById("boton-restablecer");

const campoLargoSepalo = document.getElementById("largo-sepalo");
const campoAnchoSepalo = document.getElementById("ancho-sepalo");
const campoLargoPetalo = document.getElementById("largo-petalo");
const campoAnchoPetalo = document.getElementById("ancho-petalo");

let modelo = null;
let media = null;
let desviacion = null;
let especies = null;
let ultimosValores = null;

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
  // "?t=" evita que el navegador use una copia vieja guardada en caché —
  // importante después de una corrección, para traer los pesos nuevos.
  const respuesta = await fetch(`modelo_web/pesos_modelo.json?t=${Date.now()}`);
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
  feedback.hidden = true;
  feedbackCorreccion.hidden = true;
  feedbackMensaje.hidden = true;

  if (valores.some((v) => Number.isNaN(v))) {
    mostrarError("Completa las 4 medidas con números.");
    return;
  }

  try {
    const { especie, confianza } = await predecir(valores);
    const claseEspecie = "especie-" + especie.toLowerCase();
    resultadoEspecie.textContent = especie;
    resultadoEspecie.className = "resultado-especie " + claseEspecie;
    resultado.className = "resultado " + claseEspecie;
    resultadoConfianza.textContent = `Confianza: ${(confianza * 100).toFixed(1)}%`;
    resultado.hidden = false;

    // Se guardan para poder corregir el modelo con estos mismos datos si
    // el usuario marca la predicción como equivocada.
    ultimosValores = valores;
    feedback.hidden = false;
  } catch (error) {
    mostrarError(error.message || "Ocurrió un error al predecir.");
  }
}

// Le manda la corrección al servidor: ajusta el modelo del lado de Python
// (con model.fit(), unos pocos pasos) y guarda los pesos nuevos en
// modelo_web/pesos_modelo.json. Por eso el arreglo lo ve cualquier
// visitante, no solo quien corrigió.
async function corregirModelo(especieCorrecta) {
  const respuesta = await fetch("/api/corregir", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ valores: ultimosValores, especie: especieCorrecta }),
  });

  if (!respuesta.ok) {
    const datos = await respuesta.json().catch(() => ({}));
    throw new Error(datos.error || "No se pudo guardar la corrección en el servidor.");
  }

  const { guardado_en_github: guardadoEnGithub } = await respuesta.json();

  // Se recarga el modelo ya actualizado y se vuelve a predecir con los
  // mismos datos, para mostrar en el momento que quedó corregido.
  await cargarModelo();
  await predecirYMostrar(ultimosValores);

  botonRestablecer.hidden = false;
  feedbackMensaje.textContent = guardadoEnGithub
    ? `✅ Corregido a ${especieCorrecta} y guardado en GitHub (permanente, incluso si el servidor se reinicia).`
    : `✅ Corregido a ${especieCorrecta} y guardado para todos los visitantes (mientras el servidor siga prendido).`;
  feedbackMensaje.hidden = false;
}

formulario.addEventListener("submit", (evento) => {
  evento.preventDefault();
  predecirYMostrar(leerMedidas());
});

function llenarFormulario(valores) {
  campoLargoSepalo.value = valores[0];
  campoAnchoSepalo.value = valores[1];
  campoLargoPetalo.value = valores[2];
  campoAnchoPetalo.value = valores[3];
}

// Los botones de "Probar con un ejemplo" llenan el formulario con esos 4
// números y predicen de una vez.
document.querySelectorAll(".boton-ejemplo").forEach((boton) => {
  boton.addEventListener("click", () => {
    const valores = boton.dataset.valores.split(",").map(Number);
    llenarFormulario(valores);
    predecirYMostrar(valores);
  });
});

// Botón "👍 Sí": la predicción estaba bien, no hay nada que corregir.
botonAcerto.addEventListener("click", () => {
  feedback.hidden = true;
});

// Botón "👎 No": muestra los 3 botones para elegir cuál era la especie
// correcta.
botonFallo.addEventListener("click", () => {
  feedback.hidden = true;
  feedbackCorreccion.hidden = false;
});

document.querySelectorAll(".boton-correccion").forEach((boton) => {
  boton.addEventListener("click", async () => {
    document.querySelectorAll(".boton-correccion").forEach((b) => (b.disabled = true));
    try {
      await corregirModelo(boton.dataset.especie);
    } catch (error) {
      mostrarError(error.message || "No se pudo corregir el modelo.");
    } finally {
      document.querySelectorAll(".boton-correccion").forEach((b) => (b.disabled = false));
    }
  });
});

// Le pide al servidor que vuelva a los pesos originales (los que salieron
// de entrenar_modelo.py), deshaciendo cualquier corrección guardada.
botonRestablecer.addEventListener("click", async () => {
  botonRestablecer.disabled = true;
  try {
    const respuesta = await fetch("/api/restablecer", { method: "POST" });
    if (!respuesta.ok) {
      throw new Error("No se pudo restablecer el modelo.");
    }
    const { guardado_en_github: guardadoEnGithub } = await respuesta.json();

    await cargarModelo();
    botonRestablecer.hidden = true;
    resultado.hidden = true;
    feedbackMensaje.textContent = guardadoEnGithub
      ? "↺ Modelo restablecido a los pesos originales y guardado en GitHub (permanente)."
      : "↺ Modelo restablecido a los pesos originales (para todos los visitantes, mientras el servidor siga prendido).";
    feedbackMensaje.hidden = false;
  } catch (error) {
    mostrarError(error.message || "No se pudo restablecer el modelo.");
  } finally {
    botonRestablecer.disabled = false;
  }
});

// Si se llega desde datos.html con "Usar estos datos" (que agrega
// ?ls=..&as=..&lp=..&ap=.. a la URL), se llena el formulario solo y se
// predice de una vez.
function leerMedidasDeUrl() {
  const parametros = new URLSearchParams(window.location.search);
  const claves = ["ls", "as", "lp", "ap"];
  const valores = claves.map((clave) => parseFloat(parametros.get(clave)));
  return valores.some((v) => Number.isNaN(v)) ? null : valores;
}

cargarModelo()
  .then(() => {
    estadoModelo.textContent = "Modelo listo — escribe las medidas o prueba un ejemplo.";
    botonPredecir.disabled = false;

    const valoresDeUrl = leerMedidasDeUrl();
    if (valoresDeUrl) {
      llenarFormulario(valoresDeUrl);
      predecirYMostrar(valoresDeUrl);
    }
  })
  .catch((error) => {
    estadoModelo.textContent = "No se pudo cargar el modelo.";
    mostrarError(error.message || "No se pudo cargar el modelo.");
  });
