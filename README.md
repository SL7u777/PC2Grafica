# Dibujar para hablar

Clasificador de dibujos a mano alzada para personas que perdieron el habla (afasia tras un ACV, pacientes intubados en UCI). La persona dibuja lo que necesita y la red neuronal lo traduce a una frase.

| Dibujo | Frase |
|---|---|
| vaso | Tengo sed |
| inodoro | Necesito ir al baño |
| cama | Quiero descansar |
| jeringa | Necesito mi medicina |
| celular | Quiero hablar con mi familia |

**Autores:** Isaac Antonio Martel Balvin · Sebastian Enrique Huaypar Acurio

Basado en [letras_uni](https://github.com/PeterTXS09/letras_uni).

## Flujo

1. **Recolección**: `main.py` es una app Flask con un lienzo. Pide siempre la categoría con menos dibujos y guarda cada dibujo como `datos/<categoria>/<categoria>_0001.png`, `_0002.png`, etc.
2. **Dataset**: la ruta `/prepare` recorta, centra y lleva cada dibujo a 64×64, y genera `X.npy` y `y.npy`.
3. **Entrenamiento**: `entrenamiento.ipynb` (Google Colab) entrena una CNN, muestra las métricas y al final permite clasificar cualquier imagen de `datos/` con el modelo entrenado.


## Resultados

2650 dibujos (530 por categoría), divididos en 70% entrenamiento, 15% validación y 15% prueba. Accuracy en prueba: **93.0%**.

| Categoría | Precision | Recall | F1 |
|---|---|---|---|
| cama | 0.905 | 0.950 | 0.927 |
| celular | 0.917 | 0.975 | 0.945 |
| inodoro | 0.926 | 0.938 | 0.932 |
| jeringa | 0.986 | 0.912 | 0.948 |
| vaso | 0.920 | 0.873 | 0.896 |

## Observaciones

- **Formas rectangulares**: vasos dibujados como un rectángulo alto se confunden con celular (3 casos).
- **Figuras alargadas**: jeringa → cama es el error más frecuente (5 casos); el émbolo y la aguja se parecen a las patas de una cama vista de lado.
- **Detalles curvos**: tazas con asa o con vapor se confundieron con inodoro (4 casos).
- **Cama recibe errores** de otras clases (precision más baja): ante formas "de caja", la red tiende a elegirla.
- **Jeringa es conservadora**: cuando la red la predice casi siempre acierta (precision 98.6%), pero se le escapa 1 de cada 11 (recall 91.2%).
- **Más datos, mejor resultado**: con 250 dibujos por categoría se obtuvo 88.8%; con 530, 93.0%.
- **La red aprende la forma típica**: un celular dibujado como auricular antiguo generó dudas, porque en los datos los celulares son rectángulos.
- **Seguridad**: la red puede equivocarse estando bastante segura (un inodoro tomado por celular con 68%).
- **Sin sobreajuste**: las curvas de entrenamiento y validación avanzan juntas.

## Conclusiones

- Una CNN sencilla reconoce dibujos de necesidades básicas con más de 90% de accuracy, lo que respalda el dibujo como vía de comunicación.
- No todos los errores pesan igual: confundir una jeringa con una cama ("medicina" por "descansar") es más grave que un vaso con un celular. En un uso real convendría pedir confirmación cuando la seguridad sea baja.
- Las clases con formas parecidas concentran los errores; más datos y más variados mejorarían el modelo.
- Trabajo futuro: validar con pacientes reales y agregar más necesidades (dolor, frío, ayuda).

## Uso

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Abrir http://127.0.0.1:5000, dibujar y enviar con **Enter** (borrar con **Esc**). Al terminar, visitar http://127.0.0.1:5000/prepare para generar `X.npy` y `y.npy`.
