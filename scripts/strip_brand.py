#!/usr/bin/env python3
"""Собрать поток B («Без логотипа») из потока A, стерев название бренда.

Поток A в seed-catalog.json — исходные карточки дилерского каталога.
На них написано «Алмалы» / «Алмалы-Керамик» / Almaly / Almaly Ceramic
(любой регистр). Скрипт находит эту надпись через Tesseract (rus+eng),
сверяет текст нечётким сравнением и закрашивает только полосу букв:
ровный фон восстанавливается градиентом соседних строк, фактурный —
инпейнтингом. Рисунок плитки вне этой полосы не меняется.

Запуск:
    python scripts/strip_brand.py
    python scripts/strip_brand.py --arts VHFZ612810,VHP612038
    python scripts/strip_brand.py --detect-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2
import numpy as np
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
BRANDS = (
    "АЛМАЛЫКЕРАМИК",
    "АЛМАЛЫ",
    "АЛМАЛИ",
    "ALMALYCERAMIC",
    "ALMALICERAMIC",
    "ALMALY",
    "ALMALI",
)
# Латинские буквы, которые Tesseract путает с кириллицей.
LAT2CYR = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
    }
)
# Порог принятия: ниже — не трогаем картинку. 80–86 оставляем в отчёте
# как неуверенное совпадение и не стираем.
ACCEPT_SCORE = 86.0
REVIEW_SCORE = 80.0


def normalize(text: str) -> str:
    text = text.upper().replace("Ё", "Е").translate(LAT2CYR)
    return re.sub(r"[^A-ZА-Я]", "", text)


def latin_letters(text: str) -> str:
    return re.sub(r"[^A-Z]", "", text.upper())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def ratio(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    return 100.0 * (1.0 - levenshtein(a, b) / max(len(a), len(b)))


def _best_against(candidate: str, brands: tuple[str, ...]) -> float:
    if len(candidate) < 5:
        return 0.0
    best = 0.0
    for brand in brands:
        best = max(best, ratio(candidate, brand))
        # Короткое имя часто читается с хвостом: «АЛМАЛЫ-КЕРА», «АЛМАЛЫ-КЕРАМУ».
        extra = 8 if len(brand) <= 8 else 3
        if len(brand) <= len(candidate) <= len(brand) + extra:
            best = max(best, ratio(candidate[: len(brand)], brand))
            best = max(best, ratio(candidate[-len(brand) :], brand))
        # Одна опечатка, но не обрубок («АЛМАЛ» короче имени и не должен
        # обгонять полную надпись «АЛМАЛЫ-КЕРАМИК»).
        if abs(len(candidate) - len(brand)) <= 1 and levenshtein(candidate, brand) <= 1 and len(candidate) >= 5:
            best = max(best, 90.0)
    return best


def score_brand(text: str) -> float:
    """0–100, насколько строка похожа на название бренда, а не на описание плитки.

    Латинская форма учитывается только если почти все буквы и так латинские.
    Иначе «АМ ALMALI МА МАТ» схлопывается в ALMALI и получает ложные 100.
    """
    folded = normalize(text)
    if "ГРАНИТ" in folded:
        folded = folded.replace("ГРАНИТ", "")
    best = _best_against(folded, BRANDS)
    raw = latin_letters(text)
    all_letters = re.sub(r"[^A-ZА-ЯЁ]", "", text.upper())
    if raw and len(raw) >= max(5, len(all_letters) - 1):
        if "GRANIT" in raw:
            raw = raw.replace("GRANIT", "")
        best = max(best, _best_against(raw, ("ALMALYCERAMIC", "ALMALICERAMIC", "ALMALY", "ALMALI")))
    return best


def is_ceramic_suffix(text: str) -> bool:
    folded = normalize(text)
    raw = latin_letters(text)
    if "ГРАНИТ" in folded or "GRANIT" in raw:
        return False
    if len(folded) >= 6 and ratio(folded, "КЕРАМИК") >= 78:
        return True
    if len(raw) >= 6 and ratio(raw, "CERAMIC") >= 78:
        return True
    return False


def _ocr_crop(bgr: np.ndarray, origin: tuple[int, int], psm: int) -> list[dict]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(
        rgb,
        lang="rus+eng",
        config=f"--psm {psm} --dpi 140",
        output_type=pytesseract.Output.DICT,
    )
    ox, oy = origin
    words = []
    for i, token in enumerate(data["text"]):
        token = (token or "").strip()
        if not token:
            continue
        conf = float(data["conf"][i])
        if conf < 20:
            continue
        words.append(
            {
                "x": int(data["left"][i]) + ox,
                "y": int(data["top"][i]) + oy,
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
                "conf": conf,
                "text": token,
            }
        )
    return words


def _regions(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Полный кадр плюс узкие окна сверху и снизу.

    Мелкий разрядный кикер («А Л М А Л Ы - К Е Р А М И К») теряется,
    если отдать Tesseract всю шапку: крупный заголовок перетягивает
    сегментацию. Окно ~480px по ширине его читает.
    """
    regions = [(0, 0, width, height)]
    # Выше ~160px шапка начинает включать заголовок карточки, и Tesseract
    # теряет мелкий разрядный кикер. Ниже 120px обрезается вторая строка.
    top_h = min(height, 160)
    bot_h = min(height, max(170, int(height * 0.14)))
    win, step = 460, 260
    bands = [(0, top_h), (max(0, height - bot_h), height)]
    for y0, y1 in bands:
        x = 0
        while x < width:
            x1 = min(width, x + win)
            regions.append((x, y0, x1, y1))
            if x1 == width:
                break
            x += step
    return regions


def ocr_words(bgr: np.ndarray) -> list[dict]:
    height, width = bgr.shape[:2]
    words: list[dict] = []
    for x0, y0, x1, y1 in _regions(width, height):
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        words.extend(_ocr_crop(crop, (x0, y0), psm=6))
        # Редкий разрядный «A L M A L I» в подвале psm 6 не собирает, psm 11 собирает.
        if (x1 - x0) < width or (y1 - y0) < height:
            words.extend(_ocr_crop(crop, (x0, y0), psm=11))
    return words


def _lines(words: list[dict]) -> list[list[dict]]:
    lines: list[dict] = []
    for word in sorted(words, key=lambda w: (w["y"] + w["h"] / 2, w["x"])):
        cy = word["y"] + word["h"] / 2
        placed = False
        for line in lines:
            if abs(cy - line["cy"]) <= max(8, 0.65 * max(word["h"], line["h"])):
                line["words"].append(word)
                n = len(line["words"])
                line["cy"] = (line["cy"] * (n - 1) + cy) / n
                line["h"] = max(line["h"], word["h"])
                placed = True
                break
        if not placed:
            lines.append({"cy": cy, "h": word["h"], "words": [word]})
    cleaned = []
    for line in lines:
        tokens = []
        for word in sorted(line["words"], key=lambda w: w["x"]):
            if len(normalize(word["text"])) < 2 and len(latin_letters(word["text"])) < 2:
                continue
            tokens.append(word)
        if tokens:
            cleaned.append(tokens)
    return cleaned


def find_brand_hits(words: list[dict], width: int, height: int) -> tuple[list[dict], list[dict]]:
    """Вернуть (принятые рамки, неуверенные кандидаты)."""
    accepted: list[dict] = []
    unsure: list[dict] = []
    for line in _lines(words):
        for i in range(len(line)):
            for length in range(1, 5):
                chunk = line[i : i + length]
                if len(chunk) < length:
                    break
                if length > 1:
                    gap = chunk[-1]["x"] - (chunk[-2]["x"] + chunk[-2]["w"])
                    if gap > max(70, 3.5 * max(c["h"] for c in chunk)):
                        break
                text = " ".join(c["text"] for c in chunk)
                score = score_brand(text)
                if score < REVIEW_SCORE:
                    continue
                x0 = min(c["x"] for c in chunk)
                y0 = min(c["y"] for c in chunk)
                x1 = max(c["x"] + c["w"] for c in chunk)
                y1 = max(c["y"] + c["h"] for c in chunk)
                hit = {
                    "text": text,
                    "score": round(score, 1),
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "conf": round(min(c["conf"] for c in chunk), 1),
                }
                # Дописать «Керамик», если оно стоит сразу справа отдельным словом.
                nxt = line[i + length] if i + length < len(line) else None
                if nxt is not None and is_ceramic_suffix(nxt["text"]):
                    gap = nxt["x"] - x1
                    if 0 <= gap <= max(80, 4 * hit["h"]):
                        hit["text"] = f"{text} {nxt['text']}"
                        hit["w"] = nxt["x"] + nxt["w"] - x0
                        hit["y"] = min(hit["y"], nxt["y"])
                        hit["h"] = max(y1, nxt["y"] + nxt["h"]) - hit["y"]
                        hit["score"] = round(max(score, score_brand(hit["text"])), 1)
                if hit["h"] > 88 or hit["w"] > 0.72 * width or hit["h"] > 0.09 * height:
                    hit["note"] = "рамка слишком большая, пропущена"
                    unsure.append(hit)
                    continue
                (accepted if score >= ACCEPT_SCORE else unsure).append(hit)
    accepted = _suppress_overlaps(accepted)
    return accepted, unsure


def _suppress_overlaps(hits: list[dict]) -> list[dict]:
    # При равном счёте оставляем более длинную надпись («Алмалы-Керамик», не «Алмалы»).
    def tightness(hit: dict) -> int:
        folded = normalize(hit["text"])
        if not folded:
            return 99
        return min(abs(len(folded) - len(brand)) for brand in BRANDS)

    hits = sorted(hits, key=lambda h: (-h["score"], tightness(h), -len(normalize(h["text"]))))
    kept: list[dict] = []
    for hit in hits:
        box = (hit["x"], hit["y"], hit["x"] + hit["w"], hit["y"] + hit["h"])
        area = max(1, hit["w"] * hit["h"])
        if any(_iou(box, (k["x"], k["y"], k["x"] + k["w"], k["y"] + k["h"])) > 0.35 for k in kept):
            continue
        # Отсечь почти вложенные дубли с соседних окон OCR.
        if any(_contains(box, (k["x"], k["y"], k["x"] + k["w"], k["y"] + k["h"]), area) for k in kept):
            continue
        kept.append(hit)
    return kept


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, min(area_a, area_b))


def _contains(inner, outer, inner_area: int) -> bool:
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy > 0.7 * inner_area


def _ink_columns(gray: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> tuple[np.ndarray, int]:
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.zeros(0, dtype=bool), 0
    bg = float(np.median(np.concatenate([roi[:, 0], roi[:, -1]])))
    ink = np.abs(roi.astype(np.int16) - bg) > 36
    # Буква занимает заметную долю высоты полосы, тонкая линейка — нет.
    need = max(3, int(0.30 * roi.shape[0]))
    return ink.sum(axis=0) >= need, need


def expand_spaced_word(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    """Добрать буквы, которые OCR обрезал у разрядной надписи «А Л М А Л Ы - К Е Р А М И К»."""
    height, width = gray.shape
    y0 = max(0, y0)
    y1 = min(height, y1)
    if y1 - y0 < 4:
        return x0, y0, x1, y1
    left_lim = max(0, x0 - 48)
    right_lim = min(width, x1 + 260)
    letters, _need = _ink_columns(gray, y0, y1, left_lim, right_lim)
    if letters.size == 0 or not letters.any():
        return x0, y0, x1, y1
    # Сплошная фактура пола даёт «буквы» почти в каждом столбце — тогда не расширяем.
    if float(letters.mean()) > 0.55:
        return x0, y0, x1, y1
    max_gap = max(34, int((y1 - y0) * 2.1))
    origin = x0 - left_lim
    end = x1 - left_lim

    def walk(start: int, step: int) -> int:
        x = min(max(start, 0), letters.size - 1)
        gap = 0
        last = x
        while 0 <= x < letters.size:
            if letters[x]:
                last = x
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            nxt = x + step
            if nxt < 0 or nxt >= letters.size:
                break
            x = nxt
        return last

    left = walk(origin, -1)
    right = walk(max(0, end - 1), 1)
    return left_lim + left, y0, left_lim + right + 1, y1


def tighten(gray: np.ndarray, hit: dict) -> tuple[int, int, int, int]:
    """Сжать рамку до штрихов и добрать обрезанный разрядный хвост."""
    height, width = gray.shape
    pad_x = 6
    if hit["score"] < 99:
        pad_x += max(8, int(hit["h"] * 0.9))
    x0 = max(0, hit["x"] - pad_x)
    y0 = max(0, hit["y"] - 3)
    x1 = min(width, hit["x"] + hit["w"] + 6)
    y1 = min(height, hit["y"] + hit["h"] + 3)
    x0, y0, x1, y1 = expand_spaced_word(gray, x0, y0, x1, y1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return x0, y0, x1, y1
    border = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
    bg = float(np.median(border))
    glyphs = np.abs(roi.astype(np.int16) - bg) > 16
    rows = np.where(glyphs.sum(axis=1) >= 2)[0]
    cols = np.where(glyphs.sum(axis=0) >= 2)[0]
    if len(rows) < 3 or len(cols) < 6:
        return x0, y0, x1, y1
    top = int(rows[0])
    bottom = int(rows[-1])
    left = int(cols[0])
    right = int(cols[-1])
    halo = 3
    return (
        max(0, x0 + left - halo),
        max(0, y0 + top - halo),
        min(width, x0 + right + 1 + halo),
        min(height, y0 + bottom + 1 + halo),
    )


def _gradient_fill(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    height = image.shape[0]
    above_slice = image[max(0, y0 - 8) : max(0, y0 - 1), x0:x1]
    below_slice = image[min(height, y1 + 1) : min(height, y1 + 8), x0:x1]
    if above_slice.size == 0 or below_slice.size == 0:
        return
    above = np.median(above_slice, axis=0)
    below = np.median(below_slice, axis=0)
    span = max(1, y1 - y0)
    for i, y in enumerate(range(y0, y1)):
        t = (i + 1) / (span + 1)
        image[y, x0:x1] = (1 - t) * above + t * below


def _textured(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
    height, width = image.shape[:2]
    y_a0, y_a1 = max(0, y0 - 10), max(0, y0 - 2)
    y_b0, y_b1 = min(height, y1 + 2), min(height, y1 + 10)
    samples = []
    if y_a1 > y_a0:
        samples.append(image[y_a0:y_a1, x0:x1].reshape(-1, image.shape[2]))
    if y_b1 > y_b0:
        samples.append(image[y_b0:y_b1, x0:x1].reshape(-1, image.shape[2]))
    if not samples:
        return False
    stack = np.concatenate(samples, axis=0).astype(np.float32)
    return float(stack.std(axis=0).mean()) >= 7.0


def erase_hits(image: np.ndarray, hits: list[dict]) -> np.ndarray:
    """Стереть надписи. Пиксели вне рамок возвращаются без изменений."""
    if not hits:
        return image
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = image.copy()
    mask = np.zeros(gray.shape, np.uint8)
    rects = []
    for hit in hits:
        x0, y0, x1, y1 = tighten(gray, hit)
        if x1 - x0 < 4 or y1 - y0 < 3:
            continue
        if (y1 - y0) > 90 or (x1 - x0) * (y1 - y0) > 0.05 * gray.size:
            continue
        rects.append((x0, y0, x1, y1))
        if _textured(out, x0, y0, x1, y1):
            # Сплошная полоса, не контур букв: тонкая маска оставляет читаемый ореол.
            mask[y0:y1, x0:x1] = 255
        else:
            _gradient_fill(out, x0, y0, x1, y1)
    if mask.max() > 0:
        painted = cv2.inpaint(out, mask, 3, cv2.INPAINT_TELEA)
        sel = mask > 0
        out[sel] = painted[sel]
    # Жёсткая гарантия: вне рамок кадр совпадает с исходником, включая рисунок плитки.
    keep = np.ones(gray.shape, np.uint8)
    for x0, y0, x1, y1 in rects:
        keep[y0:y1, x0:x1] = 0
    out[keep == 1] = original[keep == 1]
    return out


def still_has_brand(image: np.ndarray, hits: list[dict]) -> bool:
    height, width = image.shape[:2]
    for hit in hits:
        x0 = max(0, hit["x"] - 20)
        y0 = max(0, hit["y"] - 16)
        x1 = min(width, hit["x"] + hit["w"] + 20)
        y1 = min(height, hit["y"] + hit["h"] + 16)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        for word in _ocr_crop(crop, (0, 0), psm=6):
            if score_brand(word["text"]) >= ACCEPT_SCORE:
                return True
        # Соседние слова той же строки: «АЛМАЛЫ» + «КЕРАМИК».
        words = _ocr_crop(crop, (0, 0), psm=6)
        for line in _lines(words):
            text = " ".join(w["text"] for w in line)
            if score_brand(text) >= ACCEPT_SCORE:
                return True
    return False


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"не удалось открыть {path}")
    return image


def write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"не удалось закодировать {path}")
    path.write_bytes(buf.tobytes())


def thumbnail(image: np.ndarray, longest: int = 560) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, longest / max(height, width))
    if scale >= 1:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, cache: Path) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    dest = cache / digest
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    last_error = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "almaly-brand-clean/1.0"})
            with urllib.request.urlopen(req, timeout=40) as resp:
                tmp.write_bytes(resp.read())
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 — сеть, повторяем
            last_error = exc
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"не скачалось {url}: {last_error}")


def process_file(path: Path) -> dict:
    image = read_image(path)
    words = ocr_words(image)
    hits, unsure = find_brand_hits(words, image.shape[1], image.shape[0])
    result = {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "hits": [{k: h[k] for k in ("text", "score", "conf", "x", "y", "w", "h")} for h in hits],
        "unsure": [{k: h[k] for k in ("text", "score", "conf", "x", "y", "w", "h") if k in h} for h in unsure],
        "changed": False,
    }
    if not hits:
        result["status"] = "uncertain" if unsure else "no_brand"
        result["review"] = bool(unsure)
        return result
    cleaned = erase_hits(image, hits)
    if still_has_brand(cleaned, hits):
        # Второе прохождение: чуть шире, если ореол букв остался.
        wider = []
        for hit in hits:
            grown = dict(hit)
            grown["x"] = max(0, hit["x"] - 6)
            grown["y"] = max(0, hit["y"] - 3)
            grown["w"] = hit["w"] + 12
            grown["h"] = hit["h"] + 6
            wider.append(grown)
        cleaned = erase_hits(image, wider)
        result["widened"] = True
        if still_has_brand(cleaned, hits):
            result["status"] = "uncertain"
            result["review"] = True
            result["note"] = "после стирания OCR всё ещё видит название бренда"
        else:
            result["status"] = "removed"
            result["review"] = max(h["score"] for h in hits) < 92
    else:
        result["status"] = "removed"
        result["review"] = max(h["score"] for h in hits) < 92
    if not np.array_equal(cleaned, image):
        result["changed"] = True
        result["image"] = cleaned
    else:
        result["status"] = "uncertain"
        result["review"] = True
        result["note"] = "рамка нашлась, но пиксели не изменились"
    return result


def _public_path(art: str, filename: str) -> str:
    return f"clean/{art}/{filename}"


def _photo_record(art: str, source: dict, position: int, filename: str, rel: str, blob: bytes, tag: str) -> dict:
    return {
        "id": f"clean-{art}-{position:02d}",
        "position": position,
        "is_cover": position == 1,
        "tag": tag,
        "url": rel,
        "thumb_url": rel if filename.endswith("_t.jpg") else _public_path(art, f"{position:02d}_t.jpg"),
        "sha256": sha256_bytes(blob),
        "size_bytes": len(blob),
        "source_url": source.get("url"),
        "source_sha256": source.get("sha256") or "",
    }


def process_task(task: dict) -> dict:
    """Обработать один кадр. Вызывается и из пула процессов."""
    os.environ["OMP_THREAD_LIMIT"] = "1"
    cv2.setNumThreads(1)
    art = task["art"]
    position = task["position"]
    source = task["source"]
    out_dir = Path(task["out_dir"])
    cache = Path(task["cache"])
    detect_only = task["detect_only"]
    record = {
        "art": art,
        "name": task["name"],
        "position": position,
        "tag": source.get("tag") or ("cover" if position == 1 else "interior"),
        "source_url": source.get("url"),
        "source_sha256": source.get("sha256") or "",
    }
    try:
        path = download(source["url"], cache)
        if detect_only:
            image = read_image(path)
            words = ocr_words(image)
            hits, unsure = find_brand_hits(words, image.shape[1], image.shape[0])
            record.update(
                {
                    "status": "would_remove" if hits else ("uncertain" if unsure else "no_brand"),
                    "hits": hits,
                    "unsure": unsure,
                    "review": (not hits and bool(unsure)) or any(h["score"] < 93 for h in hits),
                }
            )
            return record
        processed = process_file(path)
        hits = processed["hits"]
        record["hits"] = hits
        record["unsure"] = processed["unsure"]
        record["status"] = processed["status"]
        record["review"] = bool(processed.get("review"))
        if processed.get("note"):
            record["note"] = processed["note"]
        if processed.get("changed") and processed["status"] in {"removed", "uncertain"}:
            image = processed["image"]
            full_name = f"{position:02d}.jpg"
            thumb_name = f"{position:02d}_t.jpg"
            full_path = out_dir / art / full_name
            thumb_path = out_dir / art / thumb_name
            write_jpeg(full_path, image, 92)
            write_jpeg(thumb_path, thumbnail(image), 84)
            blob = full_path.read_bytes()
            rel = _public_path(art, full_name)
            record["output"] = rel
            record["sha256"] = sha256_bytes(blob)
            record["size_bytes"] = len(blob)
            record["thumb"] = _public_path(art, thumb_name)
        else:
            record["output"] = source.get("url")
            record["sha256"] = source.get("sha256") or ""
            record["size_bytes"] = source.get("size_bytes") or 0
            record["thumb"] = source.get("thumb_url") or source.get("url")
            record["reused_source"] = True
    except Exception as exc:  # noqa: BLE001 — один кадр не должен ронять весь прогон
        record["status"] = "error"
        record["review"] = True
        record["note"] = str(exc)
        record["hits"] = []
        record["unsure"] = []
    return record


def build_clean_photo(record: dict, source: dict) -> dict:
    position = record["position"]
    photo = {
        "id": f"clean-{record['art']}-{position:02d}",
        "position": position,
        "is_cover": position == 1,
        "tag": record.get("tag") or source.get("tag") or "interior",
        "url": record.get("output") or source.get("url"),
        "thumb_url": record.get("thumb") or source.get("thumb_url") or source.get("url"),
        "sha256": record.get("sha256") or source.get("sha256") or "",
        "size_bytes": int(record.get("size_bytes") or source.get("size_bytes") or 0),
    }
    return photo


def summarize(records: list[dict]) -> dict:
    def count(status: str) -> int:
        return sum(1 for r in records if r.get("status") == status)

    return {
        "images": len(records),
        "removed": count("removed") + count("would_remove"),
        "no_brand": count("no_brand"),
        "uncertain": count("uncertain"),
        "errors": count("error"),
        "needs_review": sum(1 for r in records if r.get("review")),
        "models": len({r["art"] for r in records}),
    }


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Отчёт: удаление названия бренда из потока B",
        "",
        payload["summary_ru"],
        "",
        "| Артикул | Кадр | Статус | Что нашёл OCR | Проверка |",
        "|---|---|---|---|---|",
    ]
    status_ru = {
        "removed": "стёрто",
        "would_remove": "будет стёрто",
        "no_brand": "бренда нет",
        "uncertain": "нужна проверка",
        "error": "ошибка",
    }
    for rec in payload["images"]:
        if rec.get("status") == "no_brand" and not rec.get("review"):
            continue
        found = "; ".join(f"{h['text']} ({h['score']})" for h in rec.get("hits") or [])
        if not found:
            found = "; ".join(f"{h['text']} ({h['score']})?" for h in rec.get("unsure") or []) or "—"
        review = "да" if rec.get("review") else "нет"
        note = rec.get("note") or ""
        if note:
            review = f"{review}. {note}"
        lines.append(
            f"| {rec['art']} | {rec['position']:02d} {rec.get('tag','')} | {status_ru.get(rec.get('status'), rec.get('status'))} | {found} | {review} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_examples(records: list[dict], cache: Path, dest: Path, arts: list[str]) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    by_art: dict[str, list[dict]] = {}
    for rec in records:
        if rec["art"] in arts and rec.get("status") in {"removed", "uncertain", "would_remove"}:
            by_art.setdefault(rec["art"], []).append(rec)
    for art, group in by_art.items():
        group.sort(key=lambda r: r["position"])
        for rec in group[:3]:
            if not rec.get("source_url"):
                continue
            src = read_image(download(rec["source_url"], cache))
            rel = rec.get("output")
            if rel and not str(rel).startswith("http"):
                after = read_image(ROOT / rel)
            else:
                continue
            # Уменьшенная пара «до | после» целиком и кроп первой надписи.
            scale = 720 / src.shape[0]
            pair = np.concatenate(
                [
                    cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                    cv2.resize(after, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                ],
                axis=1,
            )
            full = dest / f"{art}_{rec['position']:02d}_before-after.jpg"
            write_jpeg(full, pair, 86)
            saved.append(full)
            if rec.get("hits"):
                hit = rec["hits"][0]
                pad = 36
                y0 = max(0, hit["y"] - pad)
                y1 = min(src.shape[0], hit["y"] + hit["h"] + pad)
                x0 = max(0, hit["x"] - pad)
                x1 = min(src.shape[1], hit["x"] + hit["w"] + pad)
                crop = np.concatenate([src[y0:y1, x0:x1], after[y0:y1, x0:x1]], axis=1)
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                crop_path = dest / f"{art}_{rec['position']:02d}_crop.jpg"
                write_jpeg(crop_path, crop, 90)
                saved.append(crop_path)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Стереть название Алмалы/Almaly с кадров потока A и собрать поток B")
    parser.add_argument("--catalog", type=Path, default=ROOT / "seed-catalog.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "clean")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "brand-removal.json")
    parser.add_argument("--arts", default="", help="Список артикулов через запятую. Пусто — все модели.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    parser.add_argument("--detect-only", action="store_true", help="Только отчёт OCR, файлы и каталог не менять")
    parser.add_argument("--examples", default="VHFZ612810,VHP612038,VHF612133,VHF600080", help="Артикулы для пар до/после")
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "almaly-brand-cache")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    wanted = {a.strip() for a in args.arts.split(",") if a.strip()}
    tasks = []
    for model in catalog.get("models") or []:
        art = model["art"]
        if wanted and art not in wanted:
            continue
        for index, photo in enumerate(model.get("branded_photos") or [], start=1):
            if not photo.get("url"):
                continue
            tasks.append(
                {
                    "art": art,
                    "name": model.get("name") or "",
                    "position": photo.get("position") or index,
                    "source": photo,
                    "out_dir": str(args.out_dir),
                    "cache": str(args.cache),
                    "detect_only": args.detect_only,
                }
            )
    if not tasks:
        print("Нет кадров потока A", file=sys.stderr)
        return 1

    started = time.time()
    records: list[dict] = []
    workers = 1 if len(tasks) < 3 else args.workers
    print(f"Кадров: {len(tasks)}, процессов: {workers}", flush=True)
    if workers == 1:
        for i, task in enumerate(tasks, 1):
            records.append(process_task(task))
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_task, task) for task in tasks]
            done = 0
            for future in as_completed(futures):
                records.append(future.result())
                done += 1
                if done % 20 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)}", flush=True)
    records.sort(key=lambda r: (r["art"], r["position"]))

    if not args.detect_only:
        by_key = {(r["art"], r["position"]): r for r in records}
        for model in catalog.get("models") or []:
            art = model["art"]
            if wanted and art not in wanted:
                continue
            branded = model.get("branded_photos") or []
            if not branded:
                continue
            clean = []
            for index, photo in enumerate(branded, start=1):
                position = photo.get("position") or index
                rec = by_key.get((art, position))
                if rec is None:
                    continue
                clean.append(build_clean_photo(rec, photo))
            model["clean_photos"] = clean
        catalog["brand_clean_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        args.catalog.write_text(
            json.dumps(catalog, ensure_ascii=False, separators=(", ", ": ")) + "\n",
            encoding="utf-8",
        )

    summary = summarize(records)
    summary_ru = (
        f"Моделей: {summary['models']}. Кадров потока A: {summary['images']}. "
        f"Название бренда стёрто: {summary['removed']}. "
        f"Надписи не было: {summary['no_brand']}. "
        f"Нужна ручная проверка: {summary['needs_review']}. "
        f"Ошибок скачивания/OCR: {summary['errors']}."
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline": "tesseract rus+eng + нечёткое совпадение Алмалы/Almaly + заливка фона",
        "summary": summary,
        "summary_ru": summary_ru,
        "images": records,
    }
    # В отчёт не кладём массивы пикселей — их там нет. На всякий случай уберём несериализуемое.
    for rec in records:
        rec.pop("image", None)
    write_report(args.report, payload)
    print(summary_ru, flush=True)
    print(f"Отчёт: {args.report} ({time.time() - started:.0f} с)", flush=True)

    if not args.detect_only:
        example_arts = [a.strip() for a in args.examples.split(",") if a.strip()]
        saved = save_examples(records, args.cache, ROOT / "reports" / "examples", example_arts)
        for path in saved:
            print(f"пример: {path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
