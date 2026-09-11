"""Configuración de gunicorn. post_fork() carga el modelo de TensorFlow
después de que cada worker se bifurca del proceso principal — si se carga
antes, el worker hereda hilos internos de TensorFlow ya rotos por el fork,
y el entrenamiento (model.fit()/train_on_batch()) se cuelga."""


def post_fork(server, worker):
    import app

    app.inicializar_modelo()
