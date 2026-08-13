# Firma probabilística

Demo mínima y ejecutable de **watermarking estadístico sobre el sampling** de un modelo de lenguaje.

Sin dependencias. Solo biblioteca estándar de Python.

```bash
python3 watermark_demo.py
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
    δ        z   perplejidad   detectado
  0.0     2.69         17.01   no
  0.5     4.01         17.99   sí
  1.0     6.54         18.40   sí
  2.0     9.33         19.72   sí
  4.0    20.79         30.42   sí
  8.0    23.43         42.87   sí
```

Más sesgo, más señal. Y peor texto. No hay salida elegante.

### 3. La señal crece con √N

```
 tokens        z   detectado
     20     3.61   no
     50     4.74   sí
    100     7.39   sí
    800    26.86   sí
```

Para doblar la certeza hay que cuadruplicar el texto.

### 4. Sin elección no hay dónde firmar

Cuando el modelo apenas tiene alternativas, **la marca deja de ser detectable**:

```
  mix   entropía        z   detectado
 0.30       5.20    14.03   sí
 0.80       2.87     8.94   sí
 0.95       1.89     5.67   sí
 0.99       1.52     1.88   no
```

Es el límite estructural de la técnica: una cifra, un nombre propio, una fórmula o un fragmento de código no dejan espacio donde esconder nada.

### 5. La paráfrasis no borra: diluye

```
 reescrito        z   detectado
       10%    10.25   sí
       25%     7.10   sí
       50%     3.98   no
      100%     1.86   no
```

Reescribir el 10 % apenas afecta. Hace falta reescribir la mitad para cruzar el umbral. Y lo que no se toca —cifras, nombres, términos técnicos— sigue contando verdes.

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
