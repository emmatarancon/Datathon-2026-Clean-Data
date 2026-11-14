#!/usr/bin/env python3
"""OCR v2: read the five options A-E from M3Kang images, better and faster.

Everything runs on the laptop against a fixed, deterministic sample (the
first N English rows of shard 0, the same order laptop/prepare_m3kang.py
walks with --limit). data/train.jsonl, data/val.jsonl and the shared
data/.ocr_cache.jsonl are treated as read-only: the original prepare run may
still be writing its cache while we measure. Everything this tool writes
lives in data/ocr_v2/, its own caches included.

Subcommands — measure ONE change at a time, always the same sample:
    probe     N rows of fresh OCR: rows/s, det/cls/rec breakdown, and the
              why-is-it-slow report (architecture, processes, backend).
    baseline  the existing parse, imported from prepare_m3kang, on the sample.
    run       candidate improvements, comma-separated in --improve:
                crop        OCR only the bottom --crop-frac of the image
                scale       x2 resize (cubic) before OCR
                gray        grayscale, back to 3 channels for the engine
                contrast    linear contrast, --contrast-alpha (default 1.5)
                postproc    v2 value normalization + duplicate/empty flags
                secondpass  re-OCR the failed rows with other parameters
    revise    write data/ocr_v2/revision.md + PNGs for manual comparison.
    split     write data/ocr_v2/{train,val}.jsonl (+ ocr_version) reusing the
              existing id-based split, replicated verbatim from prepare.

No new dependencies: rapidocr, cv2 and numpy are already installed (Pillow
not needed), so there is no requirements-ocr.txt and no pip.
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(ROOT, "laptop"), ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cv2                                    # noqa: E402  (opencv, already installed)
import numpy as np                            # noqa: E402
import pyarrow.parquet as pq                  # noqa: E402

import prepare_m3kang as prep                 # noqa: E402  (existing logic, unchanged)
from reason.calibration import wilson_interval  # noqa: E402

SHARD_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--qualcomm--M3Kang/snapshots/*/data/"
    "high_quality-00000-of-00004.parquet")
OUT_DIR = os.path.join(ROOT, "data", "ocr_v2")
IMAGES_DIR = os.path.join(OUT_DIR, "images")
SHARED_CACHE = os.path.join(ROOT, "data", ".ocr_cache.jsonl")  # READ-ONLY (C4)
FLAGS = ("crop", "scale", "gray", "contrast", "postproc", "secondpass")
IMAGE_FLAGS = ("crop", "scale", "gray", "contrast")

# --- the fixed sample -------------------------------------------------------

def load_sample(language="eng", limit=60):
    """The first `limit` rows of `language` in shard 0, in file order.

    Same deterministic walk as prepare_m3kang's --limit: shard rows in order,
    filtered by language. Returns (shard name, rows with image bytes).
    """
    paths = sorted(glob.glob(SHARD_GLOB))
    if not paths:
        sys.exit("shard 0 not found in the Hugging Face cache; run "
                 "laptop/prepare_m3kang.py once first")
    path = paths[0]
    wanted = [i for i, lang in enumerate(pq.read_table(path, columns=["language"]).column(0).to_pylist())
              if lang == language]
    if limit:
        wanted = wanted[:limit]
    full = pq.read_table(path, columns=["id", "language", "level", "text_only", "label", "text", "image"]).take(wanted)
    rows = []
    for i in range(full.num_rows):
        row = {c: full.column(c)[i].as_py() for c in ("id", "language", "level", "text_only", "label", "text")}
        row["image"] = full.column("image")[i].as_py()["bytes"]
        rows.append(row)
    return os.path.basename(path), rows

# --- preprocessing candidates -----------------------------------------------

def decode(image_bytes):
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)


def image_variant(flags):
    """Name of the image-level variant (what changes what the OCR sees)."""
    return "+".join(f for f in IMAGE_FLAGS if f in flags) or "plain"


def variant_id(flags, crop_frac, contrast_alpha):
    """Cache and file name: image variant + the numeric parameters, so a
    crop-frac retune never reads an older fraction's cached boxes."""
    variant = image_variant(flags)
    if "crop" in flags:
        variant += f"-cf{crop_frac:g}"
    if "contrast" in flags:
        variant += f"-ca{contrast_alpha:g}"
    return variant


def preprocess(img, flags, crop_frac, contrast_alpha):
    if "crop" in flags:
        height = img.shape[0]
        img = img[max(0, int(height * (1.0 - crop_frac))):, :]
    if "scale" in flags:
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    if "gray" in flags:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if "contrast" in flags:
        img = cv2.convertScaleAbs(img, alpha=contrast_alpha, beta=0)
    return img


def make_ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()  # same engine and defaults as prepare_m3kang

# --- caches (ONLY inside data/ocr_v2; the shared cache is read-only) --------

def cache_load(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                cache[row["key"]] = row
    return cache


def ocr_rows(ocr, rows, flags, cache_path, crop_frac, contrast_alpha, fresh=False):
    """OCR every row with this preprocessing, or load its boxes from my cache.

    The cache keeps the RAW boxes per (image variant, key), so parsing
    experiments (postproc) never repeat the OCR itself. Returns
    {key: {"boxes", "elapse", "wall", "cached"}}.
    """
    variant = variant_id(flags, crop_frac, contrast_alpha)
    cache = cache_load(cache_path)
    out = open(cache_path, "a")
    results = {}
    for n, row in enumerate(rows, 1):
        key = f"{row['id']}|{row['language']}"
        cached = cache.get(f"{variant}#{key}")
        if cached and not fresh:
            results[key] = {**cached, "cached": True}
            continue
        image = preprocess(decode(row["image"]), flags, crop_frac, contrast_alpha)
        start = time.perf_counter()
        boxes, elapse = ocr(image)
        wall = time.perf_counter() - start
        record = {"key": f"{variant}#{key}", "variant": variant, "id": row["id"],
                  "language": row["language"], "boxes": boxes or [], "wall": round(wall, 3),
                  "elapse": [round(float(e), 3) if e else 0.0 for e in (elapse or [])][:3]}
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        cache[f"{variant}#{key}"] = record
        results[key] = {**record, "cached": False}
        if n % 10 == 0:
            print(f"  [{variant}] {n}/{len(rows)} rows OCR", file=sys.stderr, flush=True)
    out.close()
    return results

# --- parsing: baseline imported, v2 = baseline + normalization + flags ------

FRACTIONS = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4",
             "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
             "⅙": "1/6", "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"}


def normalize_value(text):
    """Conservative value cleanup; fractions stay fractions.

    A drawn fraction whose bar is lost becomes "12" and no text-only pass can
    rebuild "1/2" — that is what crop and scale are for. Here: unicode
    vulgar fractions to "a/b", dashes unified, "1 / 2" -> "1/2", repeated
    spaces collapsed, stray punctuation at the ends dropped.
    """
    for glyph, slash in FRACTIONS.items():
        text = text.replace(glyph, slash)
    for dash in ("−", "–", "—"):
        text = text.replace(dash, "-")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;:,;")


def is_junk_token(token):
    """A trailing token worth dropping: punctuation-only, CJK/symbol glyphs,
    or a repeated ABCDE row. Values are never emptied by this (the caller
    keeps at least one token)."""
    if re.fullmatch(r"[A-E]{3,}", token):  # the door-label row, e.g. "ABCDE"
        return True
    if re.fullmatch(r"[^\w/]+", token) and len(token) <= 2:  # "=", "?", ")"
        return True
    non_ascii = sum(1 for ch in token if ord(ch) > 127)
    return bool(token) and non_ascii * 2 >= len(token)


def clean_tail(value):
    """Drop trailing junk tokens; return (value, stripped tokens).

    The crop keeps whatever is under the option row, and its last box often
    sticks to option E ("5 白", "E ABCDE"). A value is never emptied: at most
    it loses its trailing tokens.
    """
    tokens = value.split()
    stripped = []
    while len(tokens) > 1 and is_junk_token(tokens[-1]):
        stripped.append(tokens.pop())
    return " ".join(tokens), stripped


def parse_v2(boxes):
    """Same contract as prepare's parse_options, plus flags.

    Returns (options, dubious, notes). The assembly reuses prepare's own
    reading_order/is_marker so only the VALUES differ from the baseline.
    """
    items = prep.reading_order(boxes or [])
    starts = [i for i, item in enumerate(items)
              if prep.is_marker(item["text"], "A", strict=True) is not None]
    for start in reversed(starts):  # the options are the last A..E sequence
        options, current, i = {}, None, start
        expected = iter(prep.LETTERS)
        letter = next(expected)
        while i < len(items):
            text = items[i]["text"]
            rest = prep.is_marker(text, letter, strict=(letter == "A")) if letter else None
            if rest is not None:
                current = letter
                options[current] = normalize_value(rest)
                letter = next(expected, None)
            elif current:
                options[current] = normalize_value(options[current] + " " + text)
            i += 1
        if len(options) == 5:
            cleaned, notes = {}, []
            for k, v in options.items():
                v = v.lstrip(")）] ").strip()
                v, stripped = clean_tail(v)
                cleaned[k] = v
                if stripped:
                    notes.append(f"{k}: stripped tail {' '.join(stripped)!r}")
            if all(cleaned.values()):
                counts = Counter(cleaned.values())
                duplicated = sorted(value for value, count in counts.items() if count > 1)
                notes += [f"duplicated value: {value}" for value in duplicated]
                return cleaned, bool(duplicated), notes
            return {}, False, ["five markers but empty values: picture options"]
    return {}, False, ["no A..E sequence found"]


def is_dubious(options):
    """A five-option read is dubious when two letters share the same value.

    Rows without five options are failures, not dubious rows; the empty-value
    case cannot survive parse (both parsers reject it).
    """
    values = list((options or {}).values())
    if len(values) != 5 or any(not str(v).strip() for v in values):
        return False
    return len(set(values)) != len(values)


def assemble(sample_rows, ocr_results, flags):
    """One report row per sample row: options, dubious flag, notes, timings."""
    postproc = "postproc" in flags
    out = []
    for row in sample_rows:
        key = f"{row['id']}|{row['language']}"
        boxes = (ocr_results.get(key) or {}).get("boxes") or []
        if postproc:
            options, dubious, notes = parse_v2(boxes)
        else:
            options = prep.parse_options(boxes)
            dubious, notes = is_dubious(options), []
        secondpass = bool((ocr_results.get(key) or {}).get("secondpass"))
        if secondpass:
            notes = list(notes) + ["secondpass"]
        out.append({"key": key, "id": row["id"], "language": row["language"],
                    "level": row["level"], "text_only": row["text_only"],
                    "text": row["text"], "label": row["label"],
                    "options": options, "dubious": dubious, "notes": notes,
                    "wall": (ocr_results.get(key) or {}).get("wall")})
    return out


def parse_either(boxes, flags):
    """(options, dubious, notes) with the baseline parser or the v2 one."""
    if "postproc" in flags:
        return parse_v2(boxes)
    options = prep.parse_options(boxes)
    return options, is_dubious(options), []

# --- reporting ---------------------------------------------------------------

def load_rows_file(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def rows_path(name):
    return os.path.join(OUT_DIR, f"rows-{name}.jsonl")


def print_report(title, rows, seconds):
    n = len(rows)
    ok = [r for r in rows if len(r["options"]) == 5]
    failed = [r for r in rows if len(r["options"]) != 5]
    doubtful = [r for r in ok if r["dubious"]]
    low, high = wilson_interval(len(ok), n)
    rate = n / seconds if seconds > 0 else 0.0
    print(f"\n== {title}")
    print(f"   N {n} rows in {seconds:.1f} s ({rate:.2f} rows/s)")
    print(f"   five complete options: {len(ok)}/{n} = {100 * len(ok) / n:.1f}% "
          f"(Wilson 95%: {100 * low:.1f}%-{100 * high:.1f}%)")
    print(f"   dubious among the reads (duplicated value): {len(doubtful)}/{len(ok)}")
    print(f"   failed ({len(failed)}): " + (", ".join(r["id"] for r in failed) or "-"))
    return {"n": n, "ok": len(ok), "dubious": len(doubtful),
            "failed": [r["key"] for r in failed], "rows_s": rate}


def compare_with_baseline(rows, baseline_rows):
    base = {r["key"]: r["options"] for r in baseline_rows}
    gained, lost, changed = [], [], []
    for r in rows:
        before, after = base.get(r["key"], {}), r["options"]
        if len(after) == 5 and len(before) != 5:
            gained.append(r["id"])
        elif len(before) == 5 and len(after) != 5:
            lost.append(r["id"])
        elif before != after:
            changed.append(r["id"])
    print(f"   vs baseline: gained {len(gained)} {gained or ''}")
    print(f"   vs baseline: lost   {len(lost)} {lost or ''}")
    print(f"   vs baseline: values changed {len(changed)} {changed or ''}")

# --- subcommands -------------------------------------------------------------

def prepare_processes():
    try:
        out = subprocess.run(["pgrep", "-fl", "prepare_m3kang"],
                             capture_output=True, text=True, timeout=5)
        return [line for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        return []


def cmd_probe(args):
    shard, rows = load_sample(args.language, args.limit)
    processes = prepare_processes()
    print(f"machine: {platform.machine()} ({platform.system()}), python {platform.python_version()}")
    print(f"prepare_m3kang processes running: {len(processes)}")
    for line in processes:
        print(f"   {line.strip()}")
    if processes:
        print("   -> CONTENTION: the long OCR is running too; absolute times are "
              "inflated, relative comparisons stay valid.")
    ocr = make_ocr()
    for name in ("det_engine", "cls_engine", "rec_engine"):
        providers = getattr(getattr(ocr, name, None), "had_providers", None)
        if providers:
            print(f"{name} providers: {providers} (CoreML is available but rapidocr "
                  "1.4.4 only ever enables CPU/CUDA/DML)")
            break
    print(f"fresh OCR (no cache) on the first {len(rows)} '{args.language}' rows of {shard}")
    walls, elapses, sizes = [], [], []
    for row in rows:
        image = decode(row["image"])
        sizes.append((image.shape[0], image.shape[1]))
        start = time.perf_counter()
        _boxes, elapse = ocr(image)
        walls.append(time.perf_counter() - start)
        elapses.append([float(e) if e else 0.0 for e in (elapse or [0.0, 0.0, 0.0])][:3])
    total = sum(walls)
    print(f"rows/s: {len(rows) / total:.2f}  (mean {statistics.mean(walls):.2f} s/row, "
          f"median {statistics.median(walls):.2f} s, worst {max(walls):.2f} s)")
    det = statistics.mean(e[0] for e in elapses)
    cls = statistics.mean(e[1] for e in elapses)
    rec = statistics.mean(e[2] for e in elapses)
    model_time = max(det + cls + rec, 1e-9)
    print(f"mean per row: det {det:.2f} s ({100 * det / model_time:.0f}%), "
          f"cls {cls:.2f} s ({100 * cls / model_time:.0f}%), rec {rec:.2f} s ({100 * rec / model_time:.0f}%)")
    print(f"image size: mean {statistics.mean(h for h, _ in sizes):.0f} x "
          f"{statistics.mean(w for _, w in sizes):.0f} px; det resize rule is "
          "limit_side_len 736 with limit_type min, so large images are NOT downscaled")


def shared_cache_readonly():
    cache = {}
    if os.path.exists(SHARED_CACHE):
        with open(SHARED_CACHE) as f:
            for line in f:
                row = json.loads(line)
                cache[row["key"]] = row["options"]
    return cache


def cmd_baseline(args):
    shard, rows = load_sample(args.language, args.limit)
    os.makedirs(OUT_DIR, exist_ok=True)
    ocr = make_ocr()
    start = time.perf_counter()
    results = ocr_rows(ocr, rows, set(), os.path.join(OUT_DIR, ".cache-plain.jsonl"),
                       args.crop_frac, args.contrast_alpha)
    seconds = time.perf_counter() - start
    report_rows = assemble(rows, results, set())
    print_report(f"baseline (existing logic imported from prepare_m3kang) on {shard}",
                 report_rows, seconds)
    with open(rows_path("baseline"), "w") as f:
        for r in report_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shared = shared_cache_readonly()
    overlap = [r for r in report_rows if r["key"] in shared]
    agree = sum(1 for r in overlap if r["options"] == shared[r["key"]])
    print(f"   shared-cache cross-check (read-only): {len(overlap)}/{len(report_rows)} keys "
          f"already cached by the running prepare; {agree}/{len(overlap)} identical options")


def variant_name(flags, crop_frac=0.4, contrast_alpha=1.5):
    name = variant_id(flags, crop_frac, contrast_alpha)
    if "postproc" in flags:
        name += "+postproc"
    if "secondpass" in flags:
        name += "+2p"
    return name


def cmd_run(args):
    flags = set(f.strip() for f in args.improve.split(",") if f.strip())
    unknown = flags - set(FLAGS)
    if unknown:
        sys.exit(f"unknown flags: {', '.join(sorted(unknown))}; valid: {', '.join(FLAGS)}")
    shard, rows = load_sample(args.language, args.limit)
    os.makedirs(OUT_DIR, exist_ok=True)
    variant = variant_id(flags, args.crop_frac, args.contrast_alpha)
    cache_path = os.path.join(OUT_DIR, f".cache-{variant}.jsonl")
    ocr = make_ocr()
    start = time.perf_counter()
    results = ocr_rows(ocr, rows, flags, cache_path, args.crop_frac, args.contrast_alpha)
    if "secondpass" in flags:  # (d): only the failed rows get another pass
        first = assemble(rows, results, flags)
        failed = [row for row, r in zip(rows, first) if len(r["options"]) != 5]
        alt = (flags & set(IMAGE_FLAGS) - {"crop"})
        alt |= ({"scale"} if "crop" in flags else {"crop"}) | {"gray", "contrast"}
        if failed:
            print(f"  [secondpass] re-OCR of {len(failed)} failed rows with: "
                  f"{image_variant(alt)}", file=sys.stderr, flush=True)
            alt_results = ocr_rows(ocr, failed, alt, cache_path, args.crop_frac, args.contrast_alpha)
            for r in first:
                if len(r["options"]) != 5 and (alt_results.get(r["key"]) or {}).get("boxes"):
                    alt_boxes = alt_results[r["key"]]["boxes"]
                    alt_options, _dubious, _notes = parse_either(alt_boxes, flags)
                    if len(alt_options) == 5:
                        results[r["key"]] = {"boxes": alt_boxes,
                                             "wall": alt_results[r["key"]].get("wall"),
                                             "secondpass": True}
    report_rows = assemble(rows, results, flags)
    seconds = time.perf_counter() - start
    print_report(f"v2 variant '{variant_name(flags, args.crop_frac, args.contrast_alpha)}' on {shard}",
                 report_rows, seconds)
    baseline = load_rows_file(rows_path("baseline"))
    if baseline:
        compare_with_baseline(report_rows, baseline)
    with open(rows_path(variant_name(flags, args.crop_frac, args.contrast_alpha)), "w") as f:
        for r in report_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def latest_variant_path():
    candidates = [p for p in glob.glob(os.path.join(OUT_DIR, "rows-*.jsonl"))
                  if not p.endswith("rows-baseline.jsonl")]
    return max(candidates, key=os.path.getmtime) if candidates else ""


def cmd_revise(args):
    baseline = load_rows_file(rows_path("baseline"))
    if not baseline:
        sys.exit("run `baseline` first")
    path = rows_path(args.variant) if args.variant else latest_variant_path()
    variant_rows = load_rows_file(path)
    if not variant_rows:
        sys.exit(f"no rows file: {path}")
    variant_name_ = os.path.basename(path)[len("rows-"):-len(".jsonl")]
    base = {r["key"]: r for r in baseline}
    different = [r for r in variant_rows if r["options"] != base[r["key"]]["options"]]
    failed = [r for r in variant_rows if len(r["options"]) != 5]
    picks, seen = [], set()
    for pool in (different, failed, variant_rows):  # the interesting ones first
        for r in pool:
            if r["key"] not in seen and len(picks) < args.n:
                picks.append(r)
                seen.add(r["key"])
    _shard, sample = load_sample(args.language, args.limit or 60)
    images = {f"{r['id']}|{r['language']}": r["image"] for r in sample}
    os.makedirs(IMAGES_DIR, exist_ok=True)
    lines = ["# Revision: baseline OCR vs v2", "",
             f"sample: first {len(variant_rows)} eng rows of shard 0; variant `{variant_name_}`; "
             f"first {len(picks)} rows worth a human look (changed reads and failures first).",
             "Images saved in `data/ocr_v2/images/`. Mark the last column yourself.", "",
             "| id | baseline | v2 | notes | verdict (human) |",
             "| --- | --- | --- | --- | --- |"]
    for r in picks:
        filename = f"{r['id']}_{r['language']}.png"
        with open(os.path.join(IMAGES_DIR, filename), "wb") as f:
            f.write(images[r["key"]])
        notes = "; ".join(r["notes"]) or ("dubious" if r["dubious"] else "-")
        lines.append(f"| {r['id']} | {json.dumps(base[r['key']]['options'], ensure_ascii=False)} "
                     f"| {json.dumps(r['options'], ensure_ascii=False)} | {notes} |  |")
    with open(os.path.join(OUT_DIR, "revision.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote data/ocr_v2/revision.md ({len(picks)} rows) and {len(picks)} PNGs in data/ocr_v2/images/")


def split_of(problem_id, val_years, val_fraction):
    """Verbatim from prepare_m3kang.main(): same id always lands on the same side."""
    year = problem_id.split("_")[1]
    if val_years:
        return "val" if year in val_years else "train"
    bucket = int(hashlib.sha256(problem_id.encode()).hexdigest(), 16) % 1000
    return "val" if bucket < val_fraction * 1000 else "train"


def cmd_split(args):
    path = rows_path(args.variant) if args.variant else latest_variant_path()
    variant_rows = load_rows_file(path)
    if not variant_rows:
        sys.exit(f"no rows file: {path} (run `baseline`/`run` first)")
    name = os.path.basename(path)[len("rows-"):-len(".jsonl")]
    version = args.version or f"v2-{name}"
    val_years = set(y for y in args.val_years.split(",") if y)
    current = {}  # the existing split, READ-ONLY, only to verify the replication
    for side in ("train", "val"):
        with open(os.path.join(ROOT, "data", f"{side}.jsonl")) as f:
            for line in f:
                row = json.loads(line)
                current[f"{row['id']}|{row['language']}"] = side
    written, mismatches = {"train": [], "val": []}, 0
    for r in variant_rows:
        side = split_of(r["id"], val_years, args.val_fraction)
        if r["key"] in current and current[r["key"]] != side:
            mismatches += 1
        written[side].append({
            "id": r["id"], "language": r["language"], "level": r["level"],
            "year": int(r["id"].split("_")[1]), "text_only": r["text_only"],
            "text": r["text"].strip(), "options": r["options"], "label": r["label"],
            "ocr_version": version,
        })
    overlapping = sum(1 for r in variant_rows if r["key"] in current)
    expected = {"id", "language", "level", "year", "text_only", "text", "options", "label", "ocr_version"}
    bad = [row["id"] for side in written for row in written[side] if set(row) != expected]
    if mismatches or bad:
        sys.exit(f"split replication ({mismatches} mismatches) or field set ({len(bad)} bad rows) "
                 "does not match; nothing written")
    for side in ("train", "val"):
        with open(os.path.join(OUT_DIR, f"{side}.jsonl"), "w") as f:
            for row in written[side]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote data/ocr_v2/train.jsonl ({len(written['train'])} rows) and "
          f"data/ocr_v2/val.jsonl ({len(written['val'])} rows), ocr_version '{version}'")
    print(f"split replication check: {mismatches} mismatches over {overlapping} overlapping keys; "
          f"field-set check: {len(bad)} bad rows (copilot readers expect exactly those keys)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_sample(sp, limit):
        sp.add_argument("--language", default="eng")
        sp.add_argument("--limit", type=int, default=limit,
                        help="sample size; the full dataset needs explicit approval")

    sp = sub.add_parser("probe", help="rows/s + det/cls/rec breakdown + why it is slow")
    add_sample(sp, 20)
    sp.set_defaults(fn=cmd_probe)

    sp = sub.add_parser("baseline", help="existing logic, imported, on the fixed sample")
    add_sample(sp, 60)
    sp.add_argument("--crop-frac", type=float, default=0.4)
    sp.add_argument("--contrast-alpha", type=float, default=1.5)
    sp.set_defaults(fn=cmd_baseline)

    sp = sub.add_parser("run", help="candidate improvements, one change at a time")
    add_sample(sp, 60)
    sp.add_argument("--improve", required=True, help="comma-separated: " + ", ".join(FLAGS))
    sp.add_argument("--crop-frac", type=float, default=0.4)
    sp.add_argument("--contrast-alpha", type=float, default=1.5)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("revise", help="revision.md + PNGs for manual comparison")
    add_sample(sp, 60)
    sp.add_argument("--n", type=int, default=30)
    sp.add_argument("--variant", default="", help="rows-<name>.jsonl (default: the latest)")
    sp.set_defaults(fn=cmd_revise)

    sp = sub.add_parser("split", help="write data/ocr_v2/{train,val}.jsonl (+ ocr_version)")
    add_sample(sp, 60)
    sp.add_argument("--variant", default="", help="rows-<name>.jsonl (default: the latest)")
    sp.add_argument("--val-fraction", type=float, default=0.15)
    sp.add_argument("--val-years", default="")
    sp.add_argument("--version", default="", help="ocr_version value (default: v2-<variant>)")
    sp.set_defaults(fn=cmd_split)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()





