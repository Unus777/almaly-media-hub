"""Проверки сопоставления названия бренда. Картинки не нужны."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from strip_brand import is_ceramic_suffix, score_brand


def test_positive_names():
    samples = [
        "АЛМАЛЫ",
        "АЛМАЛЫ-КЕРАМИК",
        "АЛМАЛЫ - КЕРАМИК",
        "алмалы",
        "Алмалы-Керамик",
        "АЛМАЛЫ",
        "ALMALY",
        "Almaly",
        "almaly ceramic",
        "ALMALY CERAMIC",
        "Almaly-Ceramic",
        "almaliceramic",
        "ALMALICERAMIC",
        "ЛМАЛЫ",  # отвалилась первая буква
        "АЛМАЛИ",  # Ы прочитана как И
        "АЛМАЛЫКЕРАМИК",
        "АЛМАЛ 7 КЕРАМИК",
        "ALMALI",
        "Almali",
        "АЛМАЛ КЕРАМИК",
        "ЛМАЛЫ -КЕРАМИК",
        "АЛМАЛЫ-КЕРА",
        "АЛМАЛЫ-КЕРАМУ",
        "АЛМАЛЫ-КЕРАМ",
    ]
    for text in samples:
        score = score_brand(text)
        assert score >= 86, f"{text!r} → {score}"


def test_not_the_brand():
    samples = [
        "КЕРАМОГРАНИТ",
        "КЕРАМИК",
        "АЙССНОУ",
        "БЕТОН КРЕМ",
        "ФОРМАТ",
        "СТИЛЬ В КАЖДОЙ ДЕТАЛИ",
        "60 x 120",
        "МАТОВАЯ ПОВЕРХНОСТЬ",
        "CERAMIC",
        "GRANITE",
        "",
        "АЛМ",
        "—— ЕЕ SSS",
        "АМ ALMALI МА МАТ",
        "РАЗМЕР ПЛИТКИ",
    ]
    for text in samples:
        score = score_brand(text)
        assert score < 86, f"{text!r} не должен проходить, счёт {score}"


def test_suffix():
    assert is_ceramic_suffix("-КЕРАМИК")
    assert is_ceramic_suffix("КЕРАМИК")
    assert is_ceramic_suffix("CERAMIC")
    assert not is_ceramic_suffix("КЕРАМОГРАНИТ")
    assert not is_ceramic_suffix("ФОРМАТ")


if __name__ == "__main__":
    test_positive_names()
    test_not_the_brand()
    test_suffix()
    print("ok")
