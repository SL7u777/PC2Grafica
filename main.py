import os
import random
from flask import Flask, request, send_file, jsonify
from skimage import io
from skimage.transform import resize
import base64
import glob
import numpy as np

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


def nuevo_archivo(categoria):
    # nombre correlativo: vaso_0001.png, vaso_0002.png, ...
    existentes = glob.glob(os.path.join(DATOS, categoria, categoria + "_*.png"))
    numeros = [os.path.basename(f)[len(categoria) + 1:-4] for f in existentes]
    n = max([int(x) for x in numeros if x.isdigit()], default=0) + 1
    return os.path.join(DATOS, categoria, "{}_{:04d}.png".format(categoria, n))


main_html = """
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dibujar para hablar</title>
<style>
  body { margin: 0; padding: 16px; font-family: "Segoe UI", Roboto, Arial, sans-serif; background: #f5f1ee; color: #2b2b2b; }
  .tarjeta { display: inline-block; background: white; padding: 24px 36px 18px; margin-top: 10px; border-radius: 14px;
             border-top: 5px solid #8a1c1c; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08); }
  h1 { margin: 0 0 4px; font-size: 28px; color: #8a1c1c; }
  #frase { margin: 0 0 18px; font-size: 18px; font-style: italic; color: #a5524f; }
  canvas { border: 2px solid #333; border-radius: 6px; touch-action: none; cursor: crosshair; }
  button { font-size: 15px; padding: 9px 20px; margin: 14px 4px 0; border-radius: 8px; border: 1px solid #ccc;
           background: white; cursor: pointer; }
  button:hover { filter: brightness(0.94); }
  .enviar { background: #8a1c1c; border-color: #8a1c1c; color: white; }
  kbd { font-size: 11px; padding: 1px 5px; border-radius: 4px; border: 1px solid currentColor; opacity: 0.7; }
  #conteo { margin: 16px 0 0; font-size: 13px; color: #8c8c8c; }
</style>
</head>
<script>
  var mousePressed = false;
  var lastX, lastY;
  var ctx;
  var categoria = "";
  var dibujado = false;
  var enviando = false;

  function InitThis() {
      var canvas = document.getElementById('myCanvas');
      ctx = canvas.getContext("2d");

      canvas.addEventListener('pointerdown', function (e) {
          mousePressed = true;
          Draw(e.offsetX, e.offsetY, false);
      });
      canvas.addEventListener('pointermove', function (e) {
          if (mousePressed) {
              Draw(e.offsetX, e.offsetY, true);
          }
      });
      ['pointerup', 'pointerleave', 'pointercancel'].forEach(function (ev) {
          canvas.addEventListener(ev, function () { mousePressed = false; });
      });
      document.addEventListener('keydown', function (e) {
          if (e.key === 'Enter') enviar();
          if (e.key === 'Escape') clearArea();
      });

      fetch('/siguiente').then(r => r.json()).then(actualizar);
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

  function clearArea() {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      dibujado = false;
  }

  function enviar() {
      if (!dibujado || enviando) return;
      enviando = true;
      var datos = new FormData();
      datos.append('categoria', categoria);
      datos.append('myImage', document.getElementById('myCanvas').toDataURL());
      clearArea();
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
    <div align="center">
      <img src="https://upload.wikimedia.org/wikipedia/commons/f/f7/Uni-logo_transparente_granate.png" width="120"/>
    </div>
    <div align="center">
      <div class="tarjeta">
        <h1 id="mensaje">Cargando...</h1>
        <p id="frase"></p>
        <canvas id="myCanvas" width="200" height="200"></canvas>
        <br/>
        <button onclick="clearArea();">Borrar <kbd>Esc</kbd></button>
        <button class="enviar" onclick="enviar();">Enviar <kbd>Enter</kbd></button>
        <p id="conteo"></p>
      </div>
    </div>
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
