# Firma probabilística

Demo mínima y ejecutable de **watermarking estadístico sobre el sampling** de un modelo de lenguaje.

Sin dependencias. Solo biblioteca estándar de Python.

```bash
python3 watermark_demo.py
```

Eso es todo. `watermark_demo.py` **no tiene dependencias**.

Para verlo además sobre un modelo real:

```bash
pip install -r requirements.txt      # torch + transformers
python3 watermark_hf.py --model distilgpt2
```

---

## Por qué

Anthropic anunció que Claude firma el texto que genera. No con una etiqueta oculta ni con caracteres invisibles: **sesgando levemente la elección de palabras**, de modo que la firma *es* el patrón estadístico de sus decisiones.

Este repositorio implementa el mecanismo de la familia a la que pertenece esa técnica, para poder verlo funcionar y —sobre todo— **medir sus límites**.

> **Aviso.** Anthropic no ha publicado su implementación concreta. Esto reproduce el esquema descrito por Kirchenbauer et al. (2023), no necesariamente la variante que corre en producción.

---

## El mecanismo, en tres pasos

1. Una **clave secreta**, sembrada con el token previo, parte el vocabulario en lista **verde** y lista **roja**. La partición cambia en cada posición.
2. Antes de muestrear, se suma un sesgo **δ** a los *logits* de los verdes. No obliga a decir una palabra concreta: solo inclina la balanza cuando varias opciones estaban empatadas.
3. El detector, con la misma clave, rehace la partición, cuenta verdes y calcula:

```
z = (G − γN) / √(γ(1−γ)N)
```

Donde `G` son los verdes observados, `N` los tokens y `γ` la fracción verde del vocabulario. Sin conocer la clave, `G` debería rondar `γN`. El umbral habitual es **z > 4**.

No se detecta "estilo de IA". Se rechaza una hipótesis nula.

---

## Qué demuestra

### 1. Funciona

```
  sin marcar: verdes  52/228  (22.8%)   z =  -0.76   perplejidad  21.77
  marcado   : verdes 129/218  (59.2%)   z =  11.65   perplejidad  25.16
```

### 2. La detectabilidad se paga en calidad

```
2 · EL PRECIO DEL SESGO — δ contra calidad
====================================================================
      δ        z   perplejidad   detectado
    0.0     2.69         17.01   no
    0.5     4.01         17.99   sí
    1.0     6.54         18.40   sí
    2.0     9.33         19.72   sí
    4.0    20.79         30.42   sí
    8.0    23.43         42.87   sí
```

Más sesgo, más señal. Y peor texto. No hay salida elegante.

Fíjate en `δ=0`: da **z = 2.69**, no cero. Es el ruido del test con N finito. El umbral de 4 está a menos de sigma y medio de ese ruido, lo que explica por qué estos esquemas exigen bastante más texto del que la gente supone antes de afirmar nada.

### 3. La señal crece con √N

```
   tokens        z   detectado
       20     3.61   no
       50     4.46   sí
      100     6.50   sí
      200     8.69   sí
      400    11.29   sí
      800    14.57   sí
```

Para doblar la certeza hay que cuadruplicar el texto.

### 4. Sin elección no hay dónde firmar

Cuando el modelo apenas tiene alternativas, **la marca deja de ser detectable**:

```
  `mix` alto = el bigrama manda = el modelo casi no tiene alternativas.
```

Es el límite estructural de la técnica: una cifra, un nombre propio, una fórmula o un fragmento de código no dejan espacio donde esconder nada.

### 5. La paráfrasis no borra: diluye

```
   reescrito        z   detectado
         10%    10.25   sí
         25%     7.10   sí
         50%     3.98   no
         75%     2.06   no
        100%     1.86   no
```

Reescribir el 10 % apenas afecta. Hace falta reescribir la mitad para cruzar el umbral. Y lo que no se toca —cifras, nombres, términos técnicos— sigue contando verdes.

### 6. Sin la clave no hay señal

El mismo texto marcado, verificado con claves distintas:

```
  clave correcta                     z =  12.19   detectado
```

Un carácter de diferencia y la señal desaparece.

---

## La pregunta incómoda: ¿puedo detectarlo yo?

No. Y no es un descuido de implementación: **es el diseño**.

La partición verde/roja se siembra con la clave secreta y el token previo. Sin la clave no sabes qué tokens eran verdes en cada posición, así que no tienes contra qué contar. Para ti, "verde" es un subconjunto del vocabulario indistinguible de cualquier otro subconjunto arbitrario — y además cambia en cada paso.

Es el mismo principio que un mensaje cifrado: puedes sospechar que hay algo, pero sin la clave no hay nada que leer. Aquí incluso peor, porque el esquema está diseñado precisamente para que el texto parezca estadísticamente normal a ojos de quien no tiene la clave.

### ¿Y por aproximación?

Dos vías teóricas, ninguna práctica:

**Anomalías de perplejidad.** Un texto muy marcado es algo menos probable de lo que produciría el modelo limpio. Podrías medirlo… si tuvieras el modelo limpio. Y la señal es débil y ruidosa: no distingue un watermark de "este autor escribe raro".

**Ataques de distinción por consulta masiva.** Con suficientes generaciones sobre los mismos contextos podrías estimar la distribución real del modelo y detectar desviaciones sistemáticas, incluso reconstruir parcialmente las listas verdes de los contextos más frecuentes. Es una línea de investigación activa. Requiere un volumen enorme de consultas y se complica de forma explosiva según cuántos tokens previos se usen para sembrar la partición: con uno hay `|V|` particiones posibles, con cuatro la combinatoria lo hace inviable.

---

## Lo que esto implica

El esquema es **simétrico**: la misma clave firma y verifica.

De ahí se sigue algo que no es un detalle técnico:

**El proveedor del modelo es emisor y juez a la vez.** Ninguna de sus dos afirmaciones posibles es auditable desde fuera:

- *"Este texto lo generó nuestro modelo"* → hay que creerle.
- *"Este texto no lo generó"* → hay que creerle, y además esa afirmación tampoco se sostiene con la clave en la mano: un test estadístico no demuestra ausencia.

Y el detector es, además, un **oráculo peligroso**. Cada consulta respondida filtra información sobre la clave. Publicarlo abierto le da al atacante un banco de pruebas para afinar paráfrasis hasta que dejen de detectarse. Cerrarlo impide que nadie verifique nada. No hay salida buena.

### La alternativa que existe

**Watermarking de clave pública**: cualquiera puede verificar, nadie puede falsificar. Resolvería el problema de la auditabilidad. Hay investigación en marcha, pero es menos maduro y suele pagar la asimetría en robustez.

Anthropic no ha dicho qué usa. Todo apunta a un esquema simétrico.

---

## Sobre un modelo real: `watermark_hf.py`

`watermark_demo.py` enseña la aritmética con un modelo de juguete. `watermark_hf.py` es lo que se escribiría en producción: un `LogitsProcessor` de HuggingFace que intercepta los logits **antes** del muestreo.

```python
class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, particion, delta=2.0):
        self.p, self.delta = particion, delta

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            prev = int(input_ids[b, -1].item())
            verdes = self.p.ids_verdes(prev).to(scores.device)
            scores[b, verdes] += self.delta
        return scores
```

Eso es todo. Veinte líneas contando la partición.

```bash
pip install torch transformers
python3 watermark_hf.py --model distilgpt2 --delta 2.0
```

Ejecutado sobre `distilgpt2` (50.257 tokens, 12.564 verdes por posición, γ=0.25, δ=2.0):

```
SIN MARCAR  verdes   36/151  (23.8%)   z =  -0.33   no detectado
MARCADO     verdes  102/153  (66.7%)   z =  11.90   detectado

clave correcta ....... z =  11.90
clave equivocada ..... z =  -0.79
```

Y el precio se ve a simple vista en el texto generado:

```
sin marcar:  «a fundamental pilot project that brought together the
              Chicago Cultural Alliance (CCSAA), a small group of…»

marcado:     «a button, the connector was taken from "tree" to the
              "/tools" level and moved left to include the "reset…»
```

Con δ=2 sobre un modelo pequeño, el sesgo empuja donde no debería y el texto se resiente. Es el compromiso del que habla el experimento 2, en carne viva.

**Y ese es el punto.** Implementar el watermark es trivial *cuando controlas la inferencia*. La dificultad no es técnica: es que hace falta estar dentro del stack. Desde una API alojada —sin acceso a los logits— no hay forma de hacerlo, ni con fine-tuning, porque no controlas el modelo.

Existe una vía intermedia, *watermark distillation* ([Gu et al., ICML 2024](https://arxiv.org/abs/2312.04469)): entrenar un modelo para que genere texto marcado sin intervenir el decoder. Funciona con buena detectabilidad, y permite marcar modelos abiertos donde el usuario controla el sampling. Con dos límites documentados: el watermark se pierde al hacer fine-tuning sobre texto normal, y aprender marcas de baja distorsión exige muchísimas muestras. Y una consecuencia incómoda: si la marca es aprendible, es **falsificable** — un adversario puede generar texto dañino con la marca de otro.

---

## La trampa de la repetición

Sin filtrar pares `(previo, token)` repetidos, un texto repetitivo **infla el z-score** y produce falsos positivos: si un par frecuente cae en verde, suma una y otra vez. El test asume observaciones independientes, y la repetición rompe ese supuesto.

En este corpus, el texto **sin marcar** pasaba de `z = −0.76` a `z ≈ 3.5` — a un pelo del umbral.

```bash
python3 watermark_demo.py --no-dedup   # para verlo
```

Los esquemas reales aplican filtros equivalentes.

---

## Opciones

```
--delta FLOAT     fuerza del sesgo (default 2.0)
--gamma FLOAT     fracción verde del vocabulario (default 0.25)
--n INT           tokens a generar (default 300)
--key STR         clave secreta
--no-dedup        no filtrar pares repetidos
```

---

## Limitaciones honestas

- **El modelo es de juguete**: bigramas con backoff a unigrama sobre un corpus diminuto. El texto generado no tiene sentido. Da igual: lo que se demuestra es la aritmética del watermark, que es idéntica con un LLM real.
- **La "paráfrasis" está simulada** remuestreando del mismo modelo. Un parafraseador real conserva el significado; aquí solo se conserva la estructura estadística.
- **Esto no se puede montar sobre una API alojada.** Hace falta intervenir los *logits* antes del muestreo, así que solo puede implementarlo quien controla la inferencia.

---

## Y la asimetría que importa

**Encuentra la marca → hay evidencia.**
**No la encuentra → no hay prueba de que no exista.**

Un test estadístico no demuestra ausencia. Quien use esto para acusar a alguien, que empiece por ahí.

---

## Referencia

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., Goldstein, T.
*A Watermark for Large Language Models* (2023). https://arxiv.org/abs/2301.10226

## Licencia

MIT
