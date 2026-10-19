00:00–00:40	Neural networks from first principles	Perceptrons, layers, activations, loss functions and decision boundaries
00:40–01:20	How networks learn	Gradient descent, backpropagation, optimizers, overfitting and generalization
01:20–01:30	Break	
01:30–02:05	CNNs and image classification	Convolutions, receptive fields, pooling, feature hierarchies and classification
02:05–02:35	Segmentation architectures	U-Net, encoder–decoder structures, skip connections, Dice loss and evaluation metrics
02:35–02:45	Break	
02:45–03:45	Medical-imaging demonstration	Preprocessing, pretrained Swin UNETR, inference, segmentation overlays and failure analysis
03:45–04:00	Interpretation and discussion	Dice versus IoU, class imbalance, domain shift, limitations and reproducibility

I think this is a much stronger structure for the introduction. Rather than starting with equations and explaining neural networks abstractly, we can build the concepts in the order the audience needs them, then let them train a small network before explaining the mechanics that made it learn.

For a four-hour workshop aimed at people working with preclinical images, I would make the introduction follow this sequence:

1. Artificial intelligence → machine learning → deep learning: where neural networks fit in.
2. The basic building blocks: artificial neuron, perceptron, layers and convolution.
3. First practical exercise: train a tiny regression network in PyTorch.
4. Unpack the experiment: predictions, loss, forward pass, gradients, backpropagation and parameter updates.
5. Connect the foundations to image analysis: why convolutional networks are useful and how this leads to classification and segmentation.

The key pedagogical decision is to introduce the code before the full mathematical explanation. Participants first see that a model can learn from examples; then we explain what happened under the hood.

## Proposed introduction: first 60–70 minutes

1\. Onde encaixan as redes neuronais? — 8 min

- Que entendemos por IA?
- IA simbólica fronte a métodos baseados en datos.
- Aprendizaxe automática (Machine Learning, ML) como subcampo da IA.
- Aprendizaxe profunda (Deep Learning, DL) como subcampo de ML.
- Exemplos médicos: clasificación de imaxes, detección de lesións e segmentación de órganos.

2\. Da neurona artificial á rede neuronal — 10 min

- A neurona artificial como combinación ponderada de entradas máis un termo independente (bias).
- Funcións de activación: ReLU, sigmoide e tanh.
- O perceptrón como modelo clásico de neurona artificial e clasificador lineal.
- Capas de entrada, ocultas e saída.
- Por que necesitamos non linearidades para representar funcións complexas.

3\. Da rede densa á convolución — 7 min

- Que información hai nunha imaxe: píxeles, intensidade, canles e dimensións.
- A idea da convolución: aplicar filtros locais que detectan patróns.
- Campos receptivos, mapas de características e compartir pesos.
- Por que unha CNN é máis axeitada ca unha rede densa convencional para moitas tarefas de visión artificial.

4\. Práctica: adestrar unha rede de regresión — 15 min

- Crear datos sintéticos cunha relación coñecida.
- Definir unha rede pequena en PyTorch.
- Executar o adestramento e observar como cambia a perda.
- Representar os datos reais e as predicións do modelo.

5\. Que acabamos de facer? — 20 min

- Propagación cara adiante (forward pass).
- Predición fronte a valor real: residuo e función de perda.
- Derivadas, gradiente e regra da cadea.
- Retropropagación (backpropagation).
- Descenso do gradiente e actualización dos pesos.
- Épocas, taxa de aprendizaxe e converxencia.

## 1. Como explicaría a xerarquía da IA

A primeira diapositiva conceptual debería ser un esquema sinxelo, non unha definición longa. As redes neuronais non son sinónimo de IA: son unha familia de modelos dentro da aprendizaxe automática.

Intelixencia artificial (IA)

Sistemas que realizan tarefas asociadas á intelixencia

Aprendizaxe automática (ML)

Modelos que aprenden patróns a partir de datos

Aprendizaxe profunda (DL)

Modelos con redes neuronais de múltiples capas

Acompañaría o esquema con exemplos concretos:

| Enfoque              | Exemplo                                                       |
| -------------------- | ------------------------------------------------------------- |
| IA baseada en regras | Un sistema que aplica regras definidas por especialistas      |
| ML clásico           | Un clasificador que usa características extraídas dunha imaxe |
| DL                   | Unha CNN que aprende características directamente dos píxeles |
| DL para segmentación | Unha U-Net que predí unha etiqueta para cada píxel ou vóxel   |

Hai unha precisión importante: estes grupos non son categorías totalmente excluíntes. A aprendizaxe profunda forma parte da aprendizaxe automática, e ambas se consideran habitualmente parte da IA.

## 2. Neurona, perceptrón e convolución: canto detalle?

Non dedicaría demasiado tempo a demostrar matematicamente o funcionamento de cada compoñente. O obxectivo é que o público entenda as ideas que vai recoñecer na práctica.

[The Essential Guide to Neural Network Architectures](https://images.openai.com/static-rsc-4/m5bmlnmwYlNW5SqO1EvkssPH17OQWqJE0KeLCKlZtOfR-X5HDVd7UDcK1JLn0AtnmFGXDqUNmwWuy4o7-f1FVX2foHRf_n22G1oTjX9OfqLLHT9C6lYu62_3iE9hT0OD2FkZk9xTie0JqFgh-Rn-FhCYQ8ez24xMVf-TTxIsETk?purpose=inline)

Neurona artificial

Combina as entradas mediante pesos e suma un termo independente:

\\[ z=\sum_i w_i x_i+b,\qquad a=f(z) \\]

Os pesos controlan a importancia de cada entrada; a función de activación transforma o resultado. A rede aprende axustando os pesos e os termos independentes.

[Two-Stage Neural Network Optimization for Robust Solar Photovoltaic Forecasting | MDPI](https://images.openai.com/static-rsc-4/5GKSBwlCDoenikfzIdQsm1FlyifGa47sRbdysdlx2NpmR1rP6p6UY0_qnMIVxHLS2fTf-9x6lPhuPygujEJS0cWunTe9a4oF3jqL_6dg0THUKyqQZAiZzDj5XT7T1ZIM6TeW5pmGVr5JtBkv6I7g02jgc5RPym5k0TUc5FDtNII?purpose=inline)

Perceptrón e rede multicapa

O perceptrón clásico é un modelo de clasificación lineal baseado nunha combinación ponderada das entradas. Ao conectar unidades en capas e introducir activacións non lineais, construímos redes capaces de aproximar funcións máis complexas.

[Analytics India Magazine](https://images.openai.com/static-rsc-4/IjoKWnBmqTlC25Dna6pW5vvyVFNg6bE-SRZlXHd82fZdc4Yubhk53n61iU7-KDQFOR9jVYcO-wNNE3a_NfHQ3G3dWoDDwhq57KnvI1zNYg1y-951opwpqHxYUxaV2U-X_7yKfeCHWEUm2AI6QcL2rKNIsOQ9wQAwH6xfVYagy_M?purpose=inline)

Convolución

Un filtro pequeno percorre a imaxe e calcula respostas locais. Diferentes filtros poden aprender a responder a bordos, texturas ou estruturas máis complexas. En imaxe médica, as convolucións 3D tamén poden explotar a continuidade espacial entre vóxeles.

## 3. A práctica: que rede adestraría?

Usaría unha tarefa de regresión sobre datos sintéticos. A miña recomendación é unha función sinusoidal con algo de ruído, porque permite visualizar con claridade a diferenza entre os datos observados e as predicións, e máis adiante recuperar os conceptos de sobreaxuste e subaxuste.

Por exemplo, os datos poden seguir aproximadamente \\(y=\sin(x)\\), con pequenas perturbacións aleatorias. Unha rede con unha capa oculta pode aprender esa relación.

### Código inicial en PyTorch

Este é o tipo de código que amosaría primeiro, sen explicar aínda cada liña:

```
import torchfrom torch import nnimport matplotlib.pyplot as plttorch.manual_seed(42)# Datos sintéticos: y = sin(x) + ruídox = torch.linspace(-3.14, 3.14, 200).reshape(-1, 1)y = torch.sin(x) + 0.08 * torch.randn_like(x)# Rede neuronal pequenamodel = nn.Sequential(    nn.Linear(1, 32),    nn.ReLU(),    nn.Linear(32, 1))loss_fn = nn.MSELoss()optimizer = torch.optim.Adam(model.parameters(), lr=0.01)# Adestramentofor epoch in range(1000):    prediction = model(x)    loss = loss_fn(prediction, y)    optimizer.zero_grad()    loss.backward()    optimizer.step()# Representar os datos e a prediciónwith torch.no_grad():    prediction = model(x)plt.scatter(x.numpy(), y.numpy(), s=10, label="Datos")plt.plot(x.numpy(), prediction.numpy(), label="Rede")plt.legend()plt.show()
```

Para a sesión real, reservaría unha versión con división entre adestramento e proba (train/test split), de xeito que non se avalíe o modelo unicamente cos datos que viu. A versión anterior serve para introducir a mecánica; non debe presentarse como unha avaliación rigorosa da capacidade de xeneralización.

## 4. Despois da práctica: desmontar o adestramento

Unha vez que viron a curva achegarse aos datos, reconstruiría o proceso en cinco pasos:

## 1

Propagación cara adiante

A entrada atravesa a rede e obtense unha predición: \\(\hat{y}=f\_\theta(x)\\). Os parámetros \\(\theta\\) son os pesos e os termos independentes da rede.

## 2

Cálculo do erro

Comparamos a predición co valor real. Para regresión, un exemplo habitual é o erro cuadrático medio:

\\[ L=\frac{1}{N}\sum\_{i=1}^{N}(\hat y_i-y_i)^2 \\]

## 3

Regra da cadea

A perda depende da saída, que depende das activacións e dos parámetros de cada capa. A regra da cadea permite calcular como cambia a perda cando cambia cada parámetro.

## 4

Retropropagación

PyTorch calcula os gradientes da perda respecto dos parámetros, propagando as derivadas desde a saída cara ás capas anteriores.

## 5

Actualización dos parámetros

O optimizador usa os gradientes para modificar os parámetros e, idealmente, reducir a perda nas iteracións seguintes.

A regra da cadea merece un exemplo numérico propio. Por exemplo, se \\(z=wx+b\\), \\(a=\operatorname{ReLU}(z)\\) e \\(L=(a-y)^2\\), podemos seguir a dependencia da perda respecto de \\(w\\):

\\[ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \frac{\partial a}{\partial z} \frac{\partial z}{\partial w} \\]

Cada factor representa unha parte da dependencia. Esta expresión enlaza directamente o concepto matemático coa chamada `loss.backward()`.

## 5. Onde introducir sobreaxuste e subaxuste?

Non os explicaría como unha definición illada ao principio. Recuperaríaos despois de ver como se axusta a curva, cando o concepto de aprender unha relación xa resulte intuitivo.

- Subaxuste (underfitting): a rede é demasiado simple ou está insuficientemente adestrada para capturar o patrón; tanto o erro de adestramento coma o de validación adoitan ser altos.
- Axuste axeitado: a rede aprende a tendencia xeral e tamén funciona ben con datos non vistos.
- Sobreaxuste (overfitting): a rede adapta demasiado a súa predición aos detalles e ao ruído dos datos de adestramento; o erro de adestramento é baixo, pero o de validación é considerablemente maior.

A continuación introducimos a división adestramento/validación/proba, a regularización e a importancia de avaliar en datos independentes.

A idea central desta introdución: primeiro situar as redes neuronais no mapa da IA, despois identificar os seus compoñentes, experimentar cun modelo que aprende e finalmente explicar o mecanismo matemático da aprendizaxe. Así, cando cheguemos ás CNN, U-Net e Swin UNETR, o público xa terá unha base conceptual para entender o que fai o modelo e como interpretar os seus resultados.