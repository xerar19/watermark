#!/usr/bin/env python3
"""
Watermarking sobre un modelo real — LogitsProcessor de HuggingFace
==================================================================

El puente entre `watermark_demo.py` (modelo de juguete) y la realidad.

La demo del otro fichero enseña la aritmética. Esto es lo que se escribiría
en producción: un LogitsProcessor que intercepta los logits ANTES del
muestreo y suma δ a los tokens verdes. Son unas veinte líneas.

Ese es el punto: cuando controlas la inferencia, implementar el watermark
es trivial. La dificultad no es técnica, es que necesitas estar dentro del
stack de inferencia. Desde una API alojada no hay forma de hacerlo.

Requiere (opcional, no hace falta para watermark_demo.py):
    pip install torch transformers

Uso:
    python3 watermark_hf.py --model distilgpt2 --delta 2.0
    python3 watermark_hf.py --prompt "The network is" --tokens 200

Ejecutado sobre distilgpt2 (50.257 tokens, γ=0.25, δ=2.0):
    sin marcar    23.8% verdes   z = -0.33   no detectado
    marcado       66.7% verdes   z = 11.90   detectado
    clave errónea                z = -0.79   sin la clave, ruido
"""
from __future__ import annotations

import argparse
import math

import torch

try:
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              LogitsProcessor, LogitsProcessorList)
    HAY_TRANSFORMERS = True
except ImportError:  # permite importar el módulo sin transformers instalado
    HAY_TRANSFORMERS = False
    LogitsProcessor = object


# ─────────────────────────────────────────────────────────────────────
#  LA PARTICIÓN VERDE/ROJA
#  Sembrada con la clave y el ÚLTIMO TOKEN. Determinista: el detector la
#  reproduce exactamente. Y cambia en cada posición, así que no existe una
#  "lista de palabras de IA": existe una partición distinta en cada hueco.
# ─────────────────────────────────────────────────────────────────────

class ParticionVerde:
    def __init__(self, vocab_size: int, gamma: float = 0.25, key: int = 15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.key = key
        self.n_verdes = max(1, int(gamma * vocab_size))
        self._rng = torch.Generator(device="cpu")

    def ids_verdes(self, prev_token_id: int) -> torch.Tensor:
        """IDs del subconjunto verde para la posición que sigue a `prev_token_id`."""
        self._rng.manual_seed((self.key * (prev_token_id + 1)) % (2 ** 31 - 1))
        perm = torch.randperm(self.vocab_size, generator=self._rng, device="cpu")
        return perm[: self.n_verdes]

    def mascara_verde(self, prev_token_id: int) -> torch.Tensor:
        m = torch.zeros(self.vocab_size, dtype=torch.bool)
        m[self.ids_verdes(prev_token_id)] = True
        return m


# ─────────────────────────────────────────────────────────────────────
#  EL PROCESADOR — esto es todo lo que hay que insertar en la inferencia
# ─────────────────────────────────────────────────────────────────────

class WatermarkLogitsProcessor(LogitsProcessor):
    """
    Suma δ a los logits de los tokens verdes antes de muestrear.

    HuggingFace llama a esto en cada paso de generación, con los logits ya
    calculados y todavía sin normalizar. Aquí está el único punto donde se
    puede marcar el texto: después ya es tarde.
    """

    def __init__(self, particion: ParticionVerde, delta: float = 2.0):
        self.p = particion
        self.delta = delta

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        for b in range(input_ids.shape[0]):
            prev = int(input_ids[b, -1].item())
            verdes = self.p.ids_verdes(prev).to(scores.device)
            scores[b, verdes] += self.delta
        return scores


# ─────────────────────────────────────────────────────────────────────
#  EL DETECTOR
# ─────────────────────────────────────────────────────────────────────

def detectar(token_ids: list[int], particion: ParticionVerde,
             dedup: bool = True) -> tuple[int, int, float]:
    """
    Rehace la partición con la misma clave, cuenta verdes y calcula el z-score.

    `dedup` descarta pares (previo, token) repetidos: sin ese filtro, un texto
    repetitivo infla el z y aparecen falsos positivos, porque el test asume
    observaciones independientes.
    """
    G = N = 0
    vistos: set[tuple[int, int]] = set()
    for prev, tok in zip(token_ids, token_ids[1:]):
        if dedup:
            if (prev, tok) in vistos:
                continue
            vistos.add((prev, tok))
        if particion.mascara_verde(prev)[tok]:
            G += 1
        N += 1
    if N == 0:
        return 0, 0, 0.0
    g = particion.gamma
    z = (G - g * N) / math.sqrt(g * (1 - g) * N)
    return G, N, z


def detectar_texto(texto: str, tokenizer, particion: ParticionVerde,
                   dedup: bool = True) -> tuple[int, int, float]:
    ids = tokenizer(texto, add_special_tokens=False).input_ids
    return detectar(ids, particion, dedup)


# ─────────────────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Watermarking sobre un modelo real")
    ap.add_argument("--model", default="distilgpt2", help="modelo de HuggingFace")
    ap.add_argument("--prompt", default="The network connection was")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--key", type=int, default=15485863)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not HAY_TRANSFORMERS:
        raise SystemExit("Falta transformers:  pip install torch transformers")

    print(f"Cargando {args.model}…")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    particion = ParticionVerde(model.config.vocab_size, args.gamma, args.key)
    print(f"Vocabulario: {particion.vocab_size} tokens · "
          f"verdes por posición: {particion.n_verdes} (γ={args.gamma}) · δ={args.delta}\n")

    entrada = tok(args.prompt, return_tensors="pt")

    def generar(delta: float) -> tuple[str, list[int]]:
        torch.manual_seed(args.seed)
        procesadores = LogitsProcessorList()
        if delta > 0:
            procesadores.append(WatermarkLogitsProcessor(particion, delta))
        with torch.no_grad():
            salida = model.generate(
                **entrada,
                max_new_tokens=args.tokens,
                do_sample=True, top_k=0, temperature=1.0,
                logits_processor=procesadores,
                pad_token_id=tok.eos_token_id,
            )
        ids = salida[0].tolist()[entrada["input_ids"].shape[1]:]   # solo lo generado
        return tok.decode(ids), ids

    print("=" * 70)
    for etiqueta, d in (("SIN MARCAR", 0.0), ("MARCADO   ", args.delta)):
        texto, ids = generar(d)
        G, N, z = detectar(ids, particion)
        veredicto = "detectado" if z > 4 else "no detectado"
        print(f"{etiqueta}  verdes {G:4}/{N:<4} ({G/N*100:5.1f}%)   "
              f"z = {z:6.2f}   {veredicto}")
        print(f"            «{texto[:150].strip()}…»\n")

    # Sin la clave no hay señal: mismo texto, clave distinta.
    texto, ids = generar(args.delta)
    otra = ParticionVerde(particion.vocab_size, args.gamma, args.key + 1)
    _, _, z_mala = detectar(ids, otra)
    _, _, z_buena = detectar(ids, particion)
    print("=" * 70)
    print(f"clave correcta ....... z = {z_buena:6.2f}")
    print(f"clave equivocada ..... z = {z_mala:6.2f}")
    print("\nSin la clave no hay contra qué contar. El esquema es simétrico.")
    print("=" * 70)


if __name__ == "__main__":
    main()
