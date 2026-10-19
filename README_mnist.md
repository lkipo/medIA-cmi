# MNIST Introdución — Redes neuronais nunha CPU

Un caderno de demostración práctica para un curso de *Introdución á aprendizaxe automática
e ás redes neuronais*. Adestra dúas redes para recoñecer díxitos manuscritos (MNIST),
enteiramente nunha **CPU** en ~2 minutos en total:

- **`mnist_intro.ipynb`** — o caderno (un perceptrón/MLP e unha CNN, explicados a fondo,
  con diagramas).

## Que aprende o alumnado
- Que son un conxunto de datos, a división adestramento/proba, os lotes e as épocas.
- Como funciona un **perceptrón multicapa** (~97% de precisión).
- Como funciona unha **rede neuronal convolucional** e por que supera o MLP (~99%).
- O bucle de adestramento: paso cara adiante → perda → retropropagación → actualización dos
  pesos.
- Como inspeccionar as predicións dun modelo e os seus erros.

## Preparación

O caderno construíuse e probouse con Python 3.9 e as versións de `requirements.txt`. Desde
esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(PyTorch instálase desde o índice de «wheels» só para CPU — non fai falta GPU. Se o
instalas desde cero, usa:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.)

## Executar

```bash
source .venv/bin/activate
jupyter notebook mnist_intro.ipynb
```

Despois executa as celas de arriba a abaixo (`Maiús + Intro`). Os datos de MNIST (~55 MB)
descárganse automaticamente na primeira execución na carpeta `data/`; xa están presentes
aquí, así que o caderno funciona sen conexión.

## Ficheiros
| Ficheiro | Función |
|----------|---------|
| `mnist_intro.ipynb` | O caderno de demostración (xa executado, coas saídas) |
| `requirements.txt`  | Versións exactas dos paquetes usados |
| `data/`             | Conxunto de datos MNIST (descargado automaticamente) |
