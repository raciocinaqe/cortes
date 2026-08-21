# -*- coding: utf-8 -*-
"""Transcreve áudio/vídeo em legendas (PT-BR).
Uso: python transcrever_legenda.py <entrada> <saida.txt> [idioma] [modelo] [estilo]

estilo:
  curta  -> frases curtas (1–3 palavras)
  bloco  -> frases longas (até ~8 palavras)
"""
from __future__ import annotations

import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CT2_FORCE_CPU", "1")


def limpar_palavra(w: str) -> str:
    return (w or "").replace("|", " ").replace("\n", " ").strip()


_CORRECOES = [
    (re.compile(r"(?i)\bum\s+curso\s*d\.?\b"), "concurso"),
    (re.compile(r"(?i)\bum\s+curso\s+de\b"), "concurso de"),
    (re.compile(r"(?i)\bcom\s+curso\b"), "concurso"),
    (re.compile(r"(?i)\bconcurso\s+d\b"), "concurso"),
    (re.compile(r"(?i)\bp\s*r\s*f\b"), "PRF"),
    (re.compile(r"(?i)\bpe\s*erre\s*efe\b"), "PRF"),
    (re.compile(r"(?i)\bn[ií]vel\s+m[eé]dio\b"), "nível médio"),
    (re.compile(r"(?i)\bvoce\b"), "você"),
    (re.compile(r"(?i)\btambem\b"), "também"),
    (re.compile(r"(?i)\bnao\b"), "não"),
    (re.compile(r"(?i)\bmao\b"), "mão"),
    (re.compile(r"(?i)\bmaca\b"), "maçã"),
    (re.compile(r"(?i)\bmass[aã]o\b"), "maçã"),
    (re.compile(r"(?i)\batencao\b"), "atenção"),
    (re.compile(r"(?i)\bcoracao\b"), "coração"),
]


def corrigir_pt(texto: str) -> str:
    t = " ".join((texto or "").split())
    if not t:
        return t
    for rx, sub in _CORRECOES:
        t = rx.sub(sub, t)
    return t.strip()


def agrupar_palavras(words, max_palavras: int, max_dur: float = 1.35):
    out = []
    lote = []
    for w in words:
        texto = limpar_palavra(getattr(w, "word", "") or "")
        if not texto:
            continue
        t0 = float(getattr(w, "start", 0.0) or 0.0)
        t1 = float(getattr(w, "end", t0) or t0)
        if t1 < t0:
            t0, t1 = t1, t0
        if not lote:
            lote = [(t0, t1, texto)]
            continue
        ini = lote[0][0]
        ultimo = lote[-1][2]
        quebra = (
            len(lote) >= max_palavras
            or (t1 - ini) >= max_dur
            or ultimo.endswith((".", "!", "?", ";", ":"))
        )
        if quebra:
            frase = corrigir_pt(" ".join(x[2] for x in lote))
            if frase:
                out.append((lote[0][0], max(lote[0][0] + 0.18, lote[-1][1]), frase))
            lote = [(t0, t1, texto)]
        else:
            lote.append((t0, t1, texto))
    if lote:
        frase = corrigir_pt(" ".join(x[2] for x in lote))
        if frase:
            out.append((lote[0][0], max(lote[0][0] + 0.18, lote[-1][1]), frase))
    return out


def fundir_correcoes_multiword(pedacos):
    if len(pedacos) < 2:
        return pedacos
    out = []
    i = 0
    while i < len(pedacos):
        fundiu = False
        for n in (3, 2):
            if i + n > len(pedacos):
                continue
            janela = pedacos[i : i + n]
            junto = corrigir_pt(" ".join(p[2] for p in janela))
            original = " ".join(p[2] for p in janela)
            if junto.lower() != original.lower() and " " not in junto.strip():
                out.append((janela[0][0], janela[-1][1], junto))
                i += n
                fundiu = True
                break
            if junto.lower() != original.lower() and len(junto.split()) < len(original.split()):
                out.append((janela[0][0], janela[-1][1], junto))
                i += n
                fundiu = True
                break
        if not fundiu:
            out.append(pedacos[i])
            i += 1
    return out


def fatiar_sem_ts(texto: str, t0: float, t1: float, max_palavras: int):
    texto = corrigir_pt(texto)
    palavras = [p for p in (texto or "").split() if p]
    if not palavras:
        return []
    if len(palavras) <= max_palavras:
        return [(t0, max(t0 + 0.25, t1), " ".join(palavras))]
    dur = max(0.25, t1 - t0)
    out = []
    n = len(palavras)
    i = 0
    while i < n:
        pedaco = palavras[i : i + max_palavras]
        a = i / n
        b = min(1.0, (i + len(pedaco)) / n)
        out.append((t0 + dur * a, t0 + dur * b, " ".join(pedaco)))
        i += max_palavras
    return out


def escrever(path: str, linhas: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "uso: transcrever_legenda.py <entrada> <saida.txt> [idioma] [modelo] [estilo]",
            file=sys.stderr,
        )
        return 2
    src, dest = sys.argv[1], sys.argv[2]
    idioma = sys.argv[3] if len(sys.argv) > 3 else "pt"
    modelo = sys.argv[4] if len(sys.argv) > 4 else "medium"
    estilo = (sys.argv[5] if len(sys.argv) > 5 else "curta").lower().strip()
    if estilo in ("capcut", "curta", "frase"):
        estilo = "curta"
    elif estilo != "bloco":
        estilo = "curta"

    max_palavras = 3 if estilo == "curta" else 8
    max_dur = 1.25 if estilo == "curta" else 3.5

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        escrever(dest, ["ERRO", f"faster-whisper nao instalado: {e}"])
        return 1

    try:
        model = WhisperModel(modelo, device="cpu", compute_type="int8")
        kwargs = dict(
            language=idioma if idioma != "auto" else None,
            task="transcribe",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=280),
            beam_size=5,
            best_of=5,
            patience=1.0,
            temperature=[0.0, 0.2],
            condition_on_previous_text=True,
            initial_prompt=(
                "Vídeo em português do Brasil sobre concurso público, PRF, "
                "nível médio, salário e edital. Palavras: concurso, PRF, "
                "polícia, edital, vaga, aprovação, maçã."
            ),
            word_timestamps=True,
        )
        try:
            segments, _info = model.transcribe(
                src,
                hotwords="concurso PRF polícia edital nível médio aprovação",
                **kwargs,
            )
        except TypeError:
            segments, _info = model.transcribe(src, **kwargs)

        linhas = ["OK"]
        for s in segments:
            words = list(getattr(s, "words", None) or [])
            if words:
                pedacos = agrupar_palavras(words, max_palavras, max_dur)
                pedacos = fundir_correcoes_multiword(pedacos)
            else:
                texto = corrigir_pt(" ".join(((s.text or "").strip()).split()))
                if not texto:
                    continue
                pedacos = fatiar_sem_ts(texto, float(s.start), float(s.end), max_palavras)

            for t0, t1, tx in pedacos:
                tx = limpar_palavra(corrigir_pt(tx))
                if not tx:
                    continue
                if estilo == "curta":
                    tx = tx.upper()
                linhas.append(f"{t0:.3f}|{t1:.3f}|{tx}")
        escrever(dest, linhas)
        return 0
    except Exception as e:
        escrever(dest, ["ERRO", str(e)])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
