#!/usr/bin/env python3
"""
Watermarking estadístico sobre el sampling — demo mínima
=========================================================

Sin dependencias: solo biblioteca estándar.

Implementa el esquema de la familia Kirchenbauer et al. (2023):

  1. Una clave secreta, sembrada con el token previo, parte el vocabulario
     en lista VERDE y lista ROJA. La partición cambia en cada posición.
  2. Antes de muestrear se suma un sesgo δ a los logits de los verdes.
  3. El detector, con la misma clave, rehace la partición, cuenta verdes
     y calcula z = (G − γN) / √(γ(1−γ)N).

Nota importante: esto NO se puede montar sobre una API alojada. Hace falta
intervenir los logits ANTES del muestreo, así que solo puede hacerlo quien
controla la inferencia. Aquí usamos un modelo de juguete (bigramas con
backoff a unigrama) para que el muestreo sea real y la aritmética idéntica
a la de un LLM.

Uso:
    python watermark_demo.py            # todos los experimentos
    python watermark_demo.py --delta 3  # cambia la fuerza del sesgo
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────────────────
#  CORPUS Y MODELO DE JUGUETE
#  Un bigrama puro sobre un corpus pequeño es casi determinista (poca
#  entropía). Lo mezclamos con el unigrama para tener incertidumbre real
#  en cada paso: `mix` controla cuánta. Eso nos permitirá demostrar que
#  sin entropía no hay dónde firmar.
# ─────────────────────────────────────────────────────────────────────

CORPUS = """
la red presenta una degradacion del servicio en la sede principal
el tunel de la tienda continua caido desde primera hora de la noche
el enlace principal muestra perdida de paquetes y latencia elevada
la sesion del protocolo no llega a establecerse tras varios intentos
el equipo de la sede remota no responde a las peticiones de estado
la caida del enlace provoca una interrupcion del servicio de datos
el analisis de los registros muestra errores de negociacion repetidos
la conectividad de nivel tres funciona pero el tunel sigue caido
el cambio de configuracion no aparece reflejado en el sistema central
la degradacion del enlace secundario afecta al trafico de la sede
el servicio de la tienda queda interrumpido durante varios minutos
la revision de la configuracion no muestra cambios recientes aplicados
el estado del equipo remoto indica una perdida de conectividad parcial
la latencia del enlace principal supera el umbral definido en el contrato
el registro de eventos recoge una secuencia de errores de autenticacion
la interrupcion del servicio coincide con el cambio de la noche anterior
el tunel se restablece tras reiniciar la negociacion del protocolo
la sede remota recupera la conectividad despues del reinicio del equipo
"""


class ToyLM:
    """Modelo de bigramas con backoff a unigrama."""

    def __init__(self, corpus: str, mix: float = 0.55):
        self.mix = mix                       # peso del bigrama frente al unigrama
        toks = corpus.split()
        self.vocab = sorted(set(toks))
        self.uni = Counter(toks)
        self.bi: dict[str, Counter] = defaultdict(Counter)
        for a, b in zip(toks, toks[1:]):
            self.bi[a][b] += 1
        self._uni_total = sum(self.uni.values())

    def dist(self, prev: str) -> dict[str, float]:
        """P(siguiente | prev), mezclando bigrama y unigrama."""
        out: dict[str, float] = {}
        bi = self.bi.get(prev)
        bi_total = sum(bi.values()) if bi else 0
        for t in self.vocab:
            p_uni = self.uni[t] / self._uni_total
            p_bi = (bi[t] / bi_total) if bi_total else 0.0
            out[t] = self.mix * p_bi + (1 - self.mix) * p_uni
        s = sum(out.values())
        return {t: p / s for t, p in out.items()}


# ─────────────────────────────────────────────────────────────────────
#  EL WATERMARK
# ─────────────────────────────────────────────────────────────────────

def green_set(prev: str, key: str, vocab: list[str], gamma: float) -> set[str]:
    """
    Partición pseudoaleatoria del vocabulario, sembrada con la clave secreta
    y el token previo. Determinista: el detector la reproduce exactamente.
    """
    h = hashlib.sha256(f"{key}|{prev}".encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    k = max(1, int(round(gamma * len(vocab))))
    return set(rng.sample(vocab, k))


def sample_next(dist: dict[str, float], greens: set[str],
                delta: float, rng: random.Random) -> str:
    """
    Muestrea el siguiente token sumando δ a los logits de los verdes.
    Con delta=0 es muestreo normal: la referencia sin marcar.
    """
    toks = list(dist)
    logits = [math.log(dist[t] + 1e-12) + (delta if t in greens else 0.0) for t in toks]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    Z = sum(exps)
    probs = [e / Z for e in exps]
    return rng.choices(toks, weights=probs, k=1)[0]


def generate(lm: ToyLM, n: int, key: str, gamma: float, delta: float,
             seed: int, start: str = "la") -> tuple[list[str], float]:
    """
    Genera n tokens. Devuelve (tokens, perplejidad bajo el modelo SIN marcar).

    La perplejidad se mide contra la distribución original: así se ve cuánta
    calidad cuesta el sesgo. Es el precio del que habla el artículo.
    """
    rng = random.Random(seed)
    out = [start]
    logp = 0.0
    for _ in range(n):
        d = lm.dist(out[-1])
        greens = green_set(out[-1], key, lm.vocab, gamma)
        tok = sample_next(d, greens, delta, rng)
        logp += math.log(d[tok] + 1e-12)     # probabilidad según el modelo limpio
        out.append(tok)
    ppl = math.exp(-logp / n)
    return out, ppl


def detect(tokens: list[str], key: str, vocab: list[str],
           gamma: float, dedup: bool = True) -> tuple[int, int, float]:
    """
    Rehace la partición con la misma clave, cuenta verdes y calcula el z-score.
    Devuelve (G, N, z).

    `dedup`: cuenta cada par (previo, token) una sola vez. Sin esto, un texto
    repetitivo infla el z-score y aparecen falsos positivos: si un par frecuente
    cae en verde, suma una y otra vez. El test asume observaciones independientes
    y la repetición rompe ese supuesto. Los esquemas reales aplican filtros
    equivalentes. Pruébalo con --no-dedup para verlo.
    """
    G = N = 0
    vistos: set[tuple[str, str]] = set()
    for prev, tok in zip(tokens, tokens[1:]):
        if dedup:
            if (prev, tok) in vistos:
                continue
            vistos.add((prev, tok))
        greens = green_set(prev, key, vocab, gamma)
        N += 1
        if tok in greens:
            G += 1
    if N == 0:
        return 0, 0, 0.0
    z = (G - gamma * N) / math.sqrt(gamma * (1 - gamma) * N)
    return G, N, z


def paraphrase(tokens: list[str], lm: ToyLM, frac: float, seed: int) -> list[str]:
    """
    Ataque: reescribe una fracción de los tokens eligiendo otro candidato
    plausible del modelo. Simula una paráfrasis que conserva el sentido pero
    cambia las palabras. Los tokens no tocados conservan su color.
    """
    rng = random.Random(seed)
    out = list(tokens)
    for i in range(1, len(out)):
        if rng.random() < frac:
            d = lm.dist(out[i - 1])
            toks = list(d)
            out[i] = rng.choices(toks, weights=[d[t] for t in toks], k=1)[0]
    return out


# ─────────────────────────────────────────────────────────────────────
#  EXPERIMENTOS
# ─────────────────────────────────────────────────────────────────────

def exp_basico(lm, key, gamma, delta, n, dedup=True):
    print("\n" + "=" * 68)
    print("1 · MARCADO vs SIN MARCAR")
    print("=" * 68)
    limpio, ppl0 = generate(lm, n, key, gamma, 0.0, seed=1)
    marcado, ppl1 = generate(lm, n, key, gamma, delta, seed=1)

    for etiqueta, toks, ppl in (("sin marcar", limpio, ppl0),
                                ("marcado   ", marcado, ppl1)):
        G, N, z = detect(toks, key, lm.vocab, gamma, dedup)
        print(f"  {etiqueta}: verdes {G:3}/{N}  ({G/N*100:4.1f}%)   "
              f"z = {z:6.2f}   perplejidad {ppl:6.2f}")

    print(f"\n  Esperado sin marca: {gamma*100:.0f}% de verdes.  Umbral habitual: z > 4")
    print(f"\n  Texto marcado (primeras 22 palabras):")
    print("   ", " ".join(marcado[:22]))


def exp_delta(lm, key, gamma, n, dedup=True):
    print("\n" + "=" * 68)
    print("2 · EL PRECIO DEL SESGO — δ contra calidad")
    print("=" * 68)
    print(f"  {'δ':>5}  {'z':>7}  {'perplejidad':>12}   detectado")
    for delta in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        toks, ppl = generate(lm, n, key, gamma, delta, seed=7)
        _, _, z = detect(toks, key, lm.vocab, gamma, dedup)
        print(f"  {delta:5.1f}  {z:7.2f}  {ppl:12.2f}   {'sí' if z > 4 else 'no'}")
    print("\n  Más sesgo, más señal. Y peor texto. No hay salida elegante.")


def exp_n(lm, key, gamma, delta, dedup=True):
    print("\n" + "=" * 68)
    print("3 · LA SEÑAL CRECE CON √N")
    print("=" * 68)
    print(f"  {'tokens':>7}  {'z':>7}   detectado")
    for n in (20, 50, 100, 200, 400, 800):
        toks, _ = generate(lm, n, key, gamma, delta, seed=3)
        _, _, z = detect(toks, key, lm.vocab, gamma, dedup)
        print(f"  {n:7}  {z:7.2f}   {'sí' if z > 4 else 'no'}")
    print("\n  Para doblar la certeza hay que cuadruplicar el texto.")


def exp_entropia(key, gamma, delta, n, dedup=True):
    print("\n" + "=" * 68)
    print("4 · SIN ELECCIÓN NO HAY DÓNDE FIRMAR")
    print("=" * 68)
    print("  `mix` alto = el bigrama manda = el modelo casi no tiene alternativas.")
    print(f"\n  {'mix':>5}  {'entropía':>9}  {'z':>7}   detectado")
    for mix in (0.30, 0.55, 0.80, 0.95, 0.99):
        lm = ToyLM(CORPUS, mix=mix)
        toks, _ = generate(lm, n, key, gamma, delta, seed=5)
        # entropía media de las distribuciones recorridas
        ents = []
        for prev in toks[:-1]:
            d = lm.dist(prev)
            ents.append(-sum(p * math.log2(p) for p in d.values() if p > 0))
        _, _, z = detect(toks, key, lm.vocab, gamma, dedup)
        print(f"  {mix:5.2f}  {sum(ents)/len(ents):9.2f}  {z:7.2f}   {'sí' if z > 4 else 'no'}")
    print("\n  Menos entropía, menos sitio donde esconder la marca.")


def exp_parafrasis(lm, key, gamma, delta, n, dedup=True):
    print("\n" + "=" * 68)
    print("5 · EL ATAQUE: PARÁFRASIS")
    print("=" * 68)
    toks, _ = generate(lm, n, key, gamma, delta, seed=11)
    _, _, z0 = detect(toks, key, lm.vocab, gamma, dedup)
    print(f"  original: z = {z0:.2f}\n")
    print(f"  {'reescrito':>10}  {'z':>7}   detectado")
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        att = paraphrase(toks, lm, frac, seed=13)
        _, _, z = detect(att, key, lm.vocab, gamma, dedup)
        print(f"  {frac*100:9.0f}%  {z:7.2f}   {'sí' if z > 4 else 'no'}")
    print("\n  No borra: diluye. Lo que no se reescribe sigue contando verdes.")


def main():
    ap = argparse.ArgumentParser(description="Demo de watermarking estadístico")
    ap.add_argument("--delta", type=float, default=2.0, help="fuerza del sesgo (default 2.0)")
    ap.add_argument("--gamma", type=float, default=0.25, help="fracción verde (default 0.25)")
    ap.add_argument("--n", type=int, default=300, help="tokens a generar (default 300)")
    ap.add_argument("--key", default="clave-secreta-del-modelo")
    ap.add_argument("--no-dedup", action="store_true",
                    help="No filtrar pares repetidos: enseña cómo la repetición infla el z")
    args = ap.parse_args()

    lm = ToyLM(CORPUS)
    dedup = not args.no_dedup
    print(f"Vocabulario: {len(lm.vocab)} palabras · γ={args.gamma} · δ={args.delta} · "
          f"dedup={'sí' if dedup else 'NO'}")

    exp_basico(lm, args.key, args.gamma, args.delta, args.n, dedup)
    exp_delta(lm, args.key, args.gamma, args.n, dedup)
    exp_n(lm, args.key, args.gamma, args.delta, dedup)
    exp_entropia(args.key, args.gamma, args.delta, args.n, dedup)
    exp_parafrasis(lm, args.key, args.gamma, args.delta, args.n, dedup)

    print("\n" + "=" * 68)
    print("Encuentra la marca → hay evidencia.")
    print("No la encuentra   → no prueba nada.")
    print("=" * 68)


if __name__ == "__main__":
    main()
