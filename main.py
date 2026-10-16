import os
import random
from flask import Flask, request, send_file, jsonify
from skimage import io
from skimage.transform import resize
import base64
import glob
from io import BytesIO
import numpy as np
from PIL import Image
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from tensorflow import keras

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
DATOS = os.environ.get("DATOS", os.path.join(BASE, "datos"))
TAM = 64

CATEGORIAS = {
    "vaso": ("un vaso", "Tengo sed"),
    "inodoro": ("un inodoro", "Necesito ir al baño"),
    "cama": ("una cama", "Quiero descansar"),
    "jeringa": ("una jeringa", "Necesito mi medicina"),
    "celular": ("un celular", "Quiero hablar con mi familia"),
}

for c in CATEGORIAS:
    os.makedirs(os.path.join(DATOS, c), exist_ok=True)

MODELO = os.path.join(BASE, "modelo.keras")
modelo = keras.models.load_model(MODELO) if os.path.exists(MODELO) else None
CLASES = sorted(CATEGORIAS)


def nuevo_archivo(categoria):
    # nombre correlativo: vaso_0001.png, vaso_0002.png, ...
    existentes = glob.glob(os.path.join(DATOS, categoria, categoria + "_*.png"))
    numeros = [os.path.basename(f)[len(categoria) + 1:-4] for f in existentes]
    n = max([int(x) for x in numeros if x.isdigit()], default=0) + 1
    return os.path.join(DATOS, categoria, "{}_{:04d}.png".format(categoria, n))


estilo = """
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Domine:wght@600;700&family=Nunito+Sans:wght@400;600;700&display=swap">
<style>
  body { margin: 0; font-family: "Nunito Sans", "Segoe UI", Arial, sans-serif; background: #f5f1ee; color: #2a1f1d; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 12px 40px;
           background: white; border-bottom: 1px solid #eadfd8; }
  .marca { display: flex; align-items: center; gap: 12px; font-family: Domine, Georgia, serif; font-size: 20px; font-weight: 700; }
  .marca img { height: 40px; }
  nav a { margin-left: 8px; padding: 7px 14px; border-radius: 8px; color: #5b4d48; text-decoration: none; }
  nav a.activo { background: #f3e5df; color: #8a1c1c; font-weight: 700; }
  main { max-width: 980px; margin: 0 auto; padding: 32px 24px; }
  .titulo { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  h1 { margin: 0; font-family: Domine, Georgia, serif; font-size: 36px; }
  .info { color: #8c7f79; font-size: 14px; }
  .frase-grande { margin: 6px 0 0; font-size: 20px; font-style: italic; color: #a5524f; }
  .tarjeta { margin-top: 22px; padding: 24px; background: white; border-radius: 14px; box-shadow: 0 6px 20px rgba(42, 31, 29, 0.06); }
  canvas { display: block; margin: 0 auto; width: 300px; height: 300px; border: 2px solid #333; border-radius: 8px;
           touch-action: none; cursor: crosshair; background: white; }
  .acciones { display: flex; justify-content: center; gap: 10px; margin-top: 16px; }
  button { font-family: inherit; font-size: 15px; padding: 10px 22px; border-radius: 8px; border: 1px solid #d6cbc5;
           background: white; cursor: pointer; }
  button:hover { filter: brightness(0.95); }
  .principal { background: #8a1c1c; border-color: #8a1c1c; color: white; font-weight: 700; }
  kbd { font-size: 11px; padding: 1px 5px; margin-left: 4px; border-radius: 4px; border: 1px solid currentColor; opacity: 0.7; }
  .conteo { margin: 16px 0 0; text-align: center; font-size: 13px; color: #8c7f79; }
  h2 { margin: 30px 0 10px; font-family: Domine, Georgia, serif; font-size: 20px; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; }
  th { text-align: left; padding: 10px 16px; font-size: 13px; color: #8c7f79; font-weight: 600; background: #faf6f3; }
  td { padding: 14px 16px; border-top: 1px solid #f0e8e3; }
  .confianza { display: flex; align-items: center; gap: 10px; }
  .pista { flex: 1; height: 8px; background: #eee6e1; border-radius: 4px; overflow: hidden; max-width: 160px; }
  .relleno { height: 100%; background: #c9b3aa; }
  .ganadora .relleno, .confianza .relleno { background: #8a1c1c; }
  .barra { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
  .barra span { width: 80px; }
  .barra .pista { max-width: none; height: 12px; }
  .barra b { width: 48px; text-align: right; font-weight: 600; }
  .aviso { margin: 12px 0 0; padding: 10px 14px; border-radius: 8px; background: #fbeee5; color: #8a4b1c; font-size: 14px; }
  .vacio { color: #8c7f79; }
</style>
"""

cabecera = """
<header>
  <div class="marca">
    <img src="https://upload.wikimedia.org/wikipedia/commons/f/f7/Uni-logo_transparente_granate.png" alt="UNI"/>
    Dibujar para hablar
  </div>
  <nav><a href="/" id="nav-recolectar">Recolectar</a><a href="/demo" id="nav-demo">Demo</a></nav>
</header>
"""

lienzo_js = """
  var mousePressed = false;
  var lastX, lastY;
  var ctx;
  var dibujado = false;

  function iniciarLienzo(alEnter) {
      var canvas = document.getElementById('myCanvas');
      ctx = canvas.getContext("2d");
      // el lienzo se ve de 300 px pero guarda 200x200, igual que los datos de entrenamiento
      function punto(e) {
          var escala = canvas.width / canvas.clientWidth;
          return [e.offsetX * escala, e.offsetY * escala];
      }
      canvas.addEventListener('pointerdown', function (e) {
          mousePressed = true;
          var p = punto(e);
          Draw(p[0], p[1], false);
      });
      canvas.addEventListener('pointermove', function (e) {
          if (mousePressed) {
              var p = punto(e);
              Draw(p[0], p[1], true);
          }
      });
      ['pointerup', 'pointerleave', 'pointercancel'].forEach(function (ev) {
          canvas.addEventListener(ev, function () { mousePressed = false; });
      });
      document.addEventListener('keydown', function (e) {
          if (e.key === 'Enter') alEnter();
          if (e.key === 'Escape') clearArea();
      });
  }

  function Draw(x, y, isDown) {
      if (isDown) {
          ctx.beginPath();
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 11;
          ctx.lineJoin = "round";
          ctx.moveTo(lastX, lastY);
          ctx.lineTo(x, y);
          ctx.closePath();
          ctx.stroke();
          dibujado = true;
      }
      lastX = x; lastY = y;
  }

  function limpiarLienzo() {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      dibujado = false;
  }
"""

main_html = """
<html>
<head>
<title>Dibujar para hablar · Recolectar</title>
""" + estilo + """
</head>
<script>
""" + lienzo_js + """
  var categoria = "";
  var enviando = false;

  function InitThis() {
      document.getElementById('nav-recolectar').className = 'activo';
      iniciarLienzo(enviar);
      fetch('/siguiente').then(r => r.json()).then(actualizar);
  }

  function clearArea() { limpiarLienzo(); }

  function enviar() {
      if (!dibujado || enviando) return;
      enviando = true;
      var datos = new FormData();
      datos.append('categoria', categoria);
      datos.append('myImage', document.getElementById('myCanvas').toDataURL());
      limpiarLienzo();
      fetch('/upload', { method: 'POST', body: datos })
          .then(r => r.json())
          .then(actualizar)
          .finally(function () { enviando = false; });
  }

  function actualizar(estado) {
      categoria = estado.categoria;
      document.getElementById('mensaje').innerHTML = 'Dibuja ' + estado.nombre;
      document.getElementById('frase').innerHTML = '"' + estado.frase + '"';
      var texto = [];
      for (var c in estado.conteos) {
          texto.push(c + ': ' + estado.conteos[c]);
      }
      document.getElementById('conteo').innerHTML = texto.join(' &middot; ');
  }
</script>
<body onload="InitThis();">
""" + cabecera + """
<main>
  <div class="titulo">
    <h1 id="mensaje">Cargando...</h1>
    <span class="info">Recolección de datos · 5 categorías</span>
  </div>
  <p class="frase-grande" id="frase"></p>
  <div class="tarjeta">
    <canvas id="myCanvas" width="200" height="200"></canvas>
    <div class="acciones">
      <button onclick="clearArea();">Borrar<kbd>Esc</kbd></button>
      <button class="principal" onclick="enviar();">Enviar<kbd>Enter</kbd></button>
    </div>
    <p class="conteo" id="conteo"></p>
  </div>
</main>
</body>
</html>
"""

demo_html = """
<html>
<head>
<title>Dibujar para hablar · Demo</title>
""" + estilo + """
</head>
<script>
""" + lienzo_js + """
  var UMBRAL = 0.7;

  function InitThis() {
      document.getElementById('nav-demo').className = 'activo';
      iniciarLienzo(reconocer);
  }

  function clearArea() {
      limpiarLienzo();
      document.getElementById('resultado').innerHTML = '<p class="vacio">Dibuja lo que necesitas y presiona Enter.</p>';
  }

  function reconocer() {
      if (!dibujado) return;
      var datos = new FormData();
      datos.append('myImage', document.getElementById('myCanvas').toDataURL());
      fetch('/predecir', { method: 'POST', body: datos }).then(r => r.json()).then(mostrar);
  }

  function barra(valor) {
      return '<div class="pista"><div class="relleno" style="width:' + Math.round(valor * 100) + '%"></div></div>';
  }

  function mostrar(r) {
      if (r.error) {
          document.getElementById('resultado').innerHTML = '<p class="aviso">' + r.error + '</p>';
          return;
      }
      var primera = r.probabilidades[0], segunda = r.probabilidades[1];
      var html = '<table><tr><th>Dibujo</th><th>Frase</th><th>Confianza</th><th>Segunda opción</th></tr>' +
                 '<tr><td><b>' + primera[0] + '</b></td><td><i>"' + r.frase + '"</i></td>' +
                 '<td><div class="confianza">' + barra(primera[1]) + Math.round(primera[1] * 100) + '%</div></td>' +
                 '<td>' + segunda[0] + ' ' + Math.round(segunda[1] * 100) + '%</td></tr></table>';
      if (primera[1] < UMBRAL) html += '<p class="aviso">Seguridad baja: conviene confirmar con el paciente.</p>';
      html += '<h2>Probabilidades</h2>';
      r.probabilidades.forEach(function (fila) {
          html += '<div class="barra' + (fila[0] === primera[0] ? ' ganadora' : '') + '"><span>' + fila[0] + '</span>' +
                  barra(fila[1]) + '<b>' + Math.round(fila[1] * 100) + '%</b></div>';
      });
      document.getElementById('resultado').innerHTML = html;
  }
</script>
<body onload="InitThis();">
""" + cabecera + """
<main>
  <div class="titulo">
    <h1>Dibuja lo que necesitas</h1>
    <span class="info">Modelo: CNN · 5 categorías · entrada 64×64</span>
  </div>
  <div class="tarjeta">
    <canvas id="myCanvas" width="200" height="200"></canvas>
    <div class="acciones">
      <button onclick="clearArea();">Borrar<kbd>Esc</kbd></button>
      <button class="principal" onclick="reconocer();">Reconocer<kbd>Enter</kbd></button>
    </div>
  </div>
  <h2>Lo que reconoció la red</h2>
  <div id="resultado"><p class="vacio">Dibuja lo que necesitas y presiona Enter.</p></div>
</main>
</body>
</html>
"""


def conteos():
    return {c: len(glob.glob(os.path.join(DATOS, c, "*.png"))) for c in CATEGORIAS}


def estado():
    n = conteos()
    minimo = min(n.values())
    categoria = random.choice([c for c in n if n[c] == minimo])
    nombre, frase = CATEGORIAS[categoria]
    return jsonify(categoria=categoria, nombre=nombre, frase=frase, conteos=n)


def normalizar(img):
    # recorta al dibujo, lo centra en un cuadrado y lo lleva a TAM x TAM
    ys, xs = np.nonzero(img)
    if len(xs) == 0:
        return np.zeros((TAM, TAM), dtype=np.uint8)
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    lado = int(max(img.shape) * 1.15)
    cuadro = np.zeros((lado, lado), dtype=np.uint8)
    y0, x0 = (lado - img.shape[0]) // 2, (lado - img.shape[1]) // 2
    cuadro[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return (resize(cuadro, (TAM, TAM), anti_aliasing=True) * 255).astype(np.uint8)


@app.route("/")
def main():
    return(main_html)


@app.route("/demo")
def demo():
    return(demo_html)


@app.route('/predecir', methods=['POST'])
def predecir():
    if modelo is None:
        return jsonify(error="Falta modelo.keras: ejecuta entrenamiento.ipynb en esta carpeta.")
    img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
    alfa = np.array(Image.open(BytesIO(base64.b64decode(img_data))))[:, :, 3]
    p = modelo.predict(normalizar(alfa)[None, ..., None] / 255.0, verbose=0)[0]
    orden = np.argsort(p)[::-1]
    categoria = CLASES[orden[0]]
    return jsonify(categoria=categoria, frase=CATEGORIAS[categoria][1],
                   probabilidades=[[CLASES[i], float(p[i])] for i in orden])


@app.route("/siguiente")
def siguiente():
    return estado()


@app.route('/upload', methods=['POST'])
def upload():
    categoria = request.form.get('categoria')
    if categoria in CATEGORIAS:
        try:
            img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
            with open(nuevo_archivo(categoria), "wb") as fh:
                fh.write(base64.b64decode(img_data))
        except Exception as err:
            print("Error:", err)
    return estado()


@app.route('/prepare', methods=['GET'])
def prepare_dataset():
    images = []
    labels = []
    for categoria in CATEGORIAS:
        filelist = glob.glob(os.path.join(DATOS, categoria, '*.png'))
        if not filelist:
            continue
        images_read = io.concatenate_images(io.imread_collection(filelist))
        images_read = images_read[:, :, :, 3]
        images.append(np.array([normalizar(img) for img in images_read]))
        labels.append(np.array([categoria] * images_read.shape[0]))
    images = np.vstack(images)
    labels = np.concatenate(labels)
    np.save(os.path.join(BASE, 'X.npy'), images)
    np.save(os.path.join(BASE, 'y.npy'), labels)
    return "OK! X.npy {}".format(images.shape)


@app.route('/X.npy', methods=['GET'])
def download_X():
    return send_file(os.path.join(BASE, 'X.npy'))


@app.route('/y.npy', methods=['GET'])
def download_y():
    return send_file(os.path.join(BASE, 'y.npy'))


if __name__ == "__main__":
    app.run()
