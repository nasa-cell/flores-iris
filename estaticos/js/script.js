// Clasificador de flores Iris — lógica del navegador
//
// Cada predicción llama a /api/predecir en el servidor, que carga
// modelo_iris.h5 con Keras y devuelve la especie. La corrección
// (/api/corregir) también habla con app.py, que ajusta ese mismo modelo
// y lo guarda para todos los visitantes.

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

let ultimosValores = null;

function mostrarError(texto) {
  mensajeError.textContent = texto;
  mensajeError.hidden = false;
}

function ocultarError() {
  mensajeError.hidden = true;
}

// Le manda las 4 medidas al servidor, que normaliza, corre modelo_iris.h5
// con Keras y devuelve la especie predicha.
async function predecir(valores) {
  const respuesta = await fetch("/api/predecir", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ valores }),
  });

  if (!respuesta.ok) {
    const datos = await respuesta.json().catch(() => ({}));
    throw new Error(datos.error || "No se pudo predecir.");
  }

  const { especie, confianza } = await respuesta.json();
  return { especie, confianza };
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

// Le manda la corrección al servidor: ajusta modelo_iris.h5 del lado de
// Python (con train_on_batch(), unos pocos pasos) y lo vuelve a guardar.
// Por eso el arreglo lo ve cualquier visitante, no solo quien corrigió.
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

  // El servidor ya tiene el modelo actualizado en memoria: alcanza con
  // volver a predecir con los mismos datos para mostrar el resultado nuevo.
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

// La predicción ahora corre en el servidor (/api/predecir), así que no hace
// falta cargar ningún modelo en el navegador antes de habilitar el botón.
estadoModelo.textContent = "Listo — escribe las medidas o prueba un ejemplo.";
botonPredecir.disabled = false;

const valoresDeUrl = leerMedidasDeUrl();
if (valoresDeUrl) {
  llenarFormulario(valoresDeUrl);
  predecirYMostrar(valoresDeUrl);
}
