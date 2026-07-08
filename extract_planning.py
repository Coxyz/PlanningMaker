#!/usr/bin/env python3
"""Extract Stéphanie Tresallet's work schedule from a planning image via minicpm-v/Ollama."""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "minicpm-v-cpu"
MAX_RETRIES = 3

PROMPT = """Your response must be a ```json code block containing a single JSON object. Nothing before. Nothing after. No sentence, no explanation, no comment.

━━━ WHAT YOU ARE READING ━━━
You are looking at a work schedule table (planning) printed on paper.

Table layout (read top to bottom):
  Row 0 : one merged cell across all day columns → contains the week number formatted as "S<N>" (e.g. "S13", "S27").
  Row 1 : column headers → "Nom" | "Méti" | "Horaire" | Lundi | Mardi | Mercredi | Jeudi | Vendredi | Samedi | Dimanche | Réalisé | Total
  Row 2 : first employee (ignore this row).
  Row 3 : TRESALLET STEPHANIE — this is the only row you must extract data from.

Inside each day cell of Row 3:
  - 0 times → rest or day off (green background).
  - 2 times stacked vertically → "HHhMM-HHhMM" on top line, "HHhMM-HHhMM" on bottom line.
  - Any other count → error.

The "Horaire" column (3rd column) on Row 3 contains the contracted weekly hours (e.g. "35h00").

━━━ STRICT EXTRACTION RULES ━━━
Stop and return an error object at the FIRST problem you encounter. Do not try to continue past an error.

[IMAGE READABILITY]
→ If the image is too blurry, rotated, or cropped so that you cannot read the table: {"error": "image unreadable"}
→ If the "Nom" column contains no employee names at all: {"error": "no employee names visible"}
→ If you cannot locate the row labeled "TRESALLET" or "STEPHANIE": {"error": "TRESALLET STEPHANIE row not found"}
→ If you cannot find a cell matching "S<N>" for the week number: {"error": "week number not found"}
→ If you cannot read the value in the Horaire column for Stéphanie's row: {"error": "weekly_hours unreadable"}

[WEEK NUMBER]
→ Must be a positive integer with no "S" prefix.
→ If you read "S13", output 13. If you read "S27", output 27.
→ If the value is not a clean integer: {"error": "week number invalid: <raw value seen>"}

[WEEKLY HOURS]
→ Must be a string in "HH:MM" format (two digits, colon, two digits).
→ Convert the "h" separator to ":". "35h00" → "35:00". "35h30" → "35:30".
→ If the value cannot be converted to this format: {"error": "weekly_hours invalid: <raw value seen>"}

[SCHEDULE — 6 DAYS]
→ You must always return exactly 6 entries: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday. No more, no less.
→ If you cannot identify all 6 day columns: {"error": "cannot identify all 6 day columns"}

[DATE LABEL]
→ Must be a string in "dd/mm" format. Two digits, slash, two digits. No year, no day name.
→ "Lundi 23 mar" → "23/03". "Mardi 24 mar" → "24/03".
→ If you cannot read a date: {"error": "date unreadable on day <position>"}
→ If the date format cannot be produced: {"error": "date_label invalid on <dd/mm>: <reason>"}

[TUESDAY — MANDATORY REST DAY]
→ Tuesday "slots" must always be null. No exceptions. Do not read the cell.

[WORKING DAYS — SLOTS]
→ Each working day cell must contain exactly 2 time slots (top = morning, bottom = afternoon).
→ If you see 0 times in a cell: "slots" must be {} (empty object — NOT null, NOT []).
→ If you see exactly 1 time: {"error": "only 1 slot on <date_label>: both morning and afternoon required"}
→ If you see more than 2 times: {"error": "more than 2 slots on <date_label>"}
→ If you cannot read one of the times clearly: {"error": "unreadable time on <date_label> <morning|afternoon>"}

[TIME FORMAT — start / end]
→ Each time must be a string in "HH:MM" format (two digits, colon, two digits, 24-hour clock).
→ Convert "h" → ":". "09h15" → "09:15". "20h00" → "20:00".
→ Allowed range: "09:15" to "20:00" inclusive.
→ "start" must be strictly earlier than "end".
→ Violations:
    - Wrong format (e.g. "9:15", "09h15", 915): {"error": "time format invalid on <date_label> <morning|afternoon>: <raw value>"}
    - Out of range: {"error": "time out of range on <date_label> <morning|afternoon>: <value>"}
    - start >= end: {"error": "start not before end on <date_label> <morning|afternoon>"}

━━━ FEW-SHOT EXAMPLES ━━━
These illustrate FORMAT only. Values are fictional. Never copy them into your answer.
Only output values you can actually read from the image.

Example A — week cell reads "S27":
  → "week": 27

Example B — Horaire cell reads "35h00":
  → "weekly_hours": "35:00"

Example C — Monday cell reads "AAhBB-CChDD" (top) and "EEhFF-GGhHH" (bottom):
  → "date_label": "dd/mm",
     "slots": {
       "morning":   {"start": "AA:BB", "end": "CC:DD"},
       "afternoon": {"start": "EE:FF", "end": "GG:HH"}
     }

Example D — Tuesday (always rest):
  → "date_label": "dd/mm", "slots": null

Example E — A working day with no times visible (green / empty cell):
  → "date_label": "dd/mm", "slots": {}

━━━ OUTPUT FORMAT ━━━
Produce exactly this structure. All 6 days must be present.

```json
{
  "week": <integer>,
  "weekly_hours": "<HH:MM>",
  "schedule": [
    {"date_label": "<dd/mm>", "slots": {"morning": {"start": "<HH:MM>", "end": "<HH:MM>"}, "afternoon": {"start": "<HH:MM>", "end": "<HH:MM>"}}},
    {"date_label": "<dd/mm>", "slots": null},
    {"date_label": "<dd/mm>", "slots": {"morning": {"start": "<HH:MM>", "end": "<HH:MM>"}, "afternoon": {"start": "<HH:MM>", "end": "<HH:MM>"}}},
    {"date_label": "<dd/mm>", "slots": {"morning": {"start": "<HH:MM>", "end": "<HH:MM>"}, "afternoon": {"start": "<HH:MM>", "end": "<HH:MM>"}}},
    {"date_label": "<dd/mm>", "slots": {"morning": {"start": "<HH:MM>", "end": "<HH:MM>"}, "afternoon": {"start": "<HH:MM>", "end": "<HH:MM>"}}},
    {"date_label": "<dd/mm>", "slots": {"morning": {"start": "<HH:MM>", "end": "<HH:MM>"}, "afternoon": {"start": "<HH:MM>", "end": "<HH:MM>"}}}
  ]
}
```"""


def save_step(img: np.ndarray, debug_dir: Path, name: str) -> None:
    path = debug_dir / f"{name}.png"
    cv2.imwrite(str(path), img)
    print(f"[debug] {name} → {path}", file=sys.stderr)


def detect_table(
    img: np.ndarray, debug_dir: Path | None = None
) -> tuple[int, int, int, int] | None:
    """Detect the main table bounding box via morphological line detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    grid = cv2.add(h_lines, v_lines)

    if debug_dir:
        save_step(grid, debug_dir, "02_table_grid_detection")

    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    merged = cv2.dilate(grid, merge_kernel)

    if debug_dir:
        save_step(merged, debug_dir, "03_table_grid_merged")

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
    x, y, w, h = cv2.boundingRect(largest)

    pad = 8
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)

    return x, y, w, h


def preprocess_image(
    image_path: str,
    crop: tuple[int, int, int, int] | None = None,
    auto_crop: bool = True,
    max_width: int = 2048,
    debug_dir: Path | None = None,
) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    if debug_dir:
        save_step(img, debug_dir, "01_original")

    if crop:
        x, y, w, h = crop
        img = img[y : y + h, x : x + w]
        if debug_dir:
            save_step(img, debug_dir, "02_crop_manual")
    elif auto_crop:
        bbox = detect_table(img, debug_dir=debug_dir)
        if bbox:
            x, y, w, h = bbox
            img = img[y : y + h, x : x + w]
            print(f"[debug] Table detected at x={x} y={y} w={w} h={h}", file=sys.stderr)
            if debug_dir:
                save_step(img, debug_dir, "04_crop_auto")
        else:
            print("[debug] Table not detected, using full image", file=sys.stderr)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug_dir:
        save_step(gray, debug_dir, "05_grayscale")

    # Binarization at fixed threshold 128 — equivalent to PIL img.point(lambda x: 0 if x < 128 else 255, '1')
    _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    if debug_dir:
        save_step(gray, debug_dir, "06_binarized")

    h, w = gray.shape
    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        gray = cv2.resize(gray, (max_width, new_h), interpolation=cv2.INTER_AREA)
        if debug_dir:
            save_step(gray, debug_dir, "06_resized")

    return gray


def encode_image(img: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buffer).decode("utf-8")


def _chat_url(generate_url: str) -> str:
    if "/api/generate" in generate_url:
        return generate_url.replace("/api/generate", "/api/chat")
    return generate_url.rstrip("/") + "/api/chat"


def chat_ollama(
    messages: list[dict],
    url: str,
    timeout: int = 600,
    keep_alive: bool = False,
) -> tuple[str, list[dict]]:
    """Send messages to Ollama /api/chat, return (response_text, updated_messages)."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "format": "json",
        "keep_alive": -1 if keep_alive else 0,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }

    response = requests.post(_chat_url(url), json=payload, timeout=timeout, stream=True)
    response.raise_for_status()

    chunks = []
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        chunk = data.get("message", {}).get("content", "")
        if chunk:
            print(chunk, end="", flush=True, file=sys.stderr)
            chunks.append(chunk)
        if data.get("done"):
            print(file=sys.stderr)
            break

    assistant_text = "".join(chunks)
    updated = messages + [{"role": "assistant", "content": assistant_text}]
    return assistant_text, updated


def parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    # Strip markdown code fences if the model added them despite instructions
    if "```" in raw:
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    # Extract the JSON object even if the model added surrounding text
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response:\n{raw}")
    return json.loads(raw[start:end])


def _parse_hhmm(value: str) -> int | None:
    """Parse 'hh:mm' into total minutes, return None on failure."""
    try:
        h, m = map(int, value.split(":"))
        return h * 60 + m
    except (ValueError, AttributeError):
        return None


def _parse_weekly_hours(value: str) -> int | None:
    """Parse '35h00' or '35:00' into total minutes, return None on failure."""
    value = str(value).strip()
    for sep in ("h", "H", ":"):
        if sep in value:
            parts = value.split(sep, 1)
            try:
                return int(parts[0]) * 60 + int(parts[1])
            except (ValueError, IndexError):
                return None
    try:
        return int(float(value) * 60)
    except ValueError:
        return None


def validate_schedule(result: dict) -> list[str]:
    issues = []

    if "error" in result:
        issues.append(f"Model returned error: {result['error']}")
        return issues

    if "week" not in result:
        issues.append("Missing 'week' field")
    if "weekly_hours" not in result:
        issues.append("Missing 'weekly_hours' field")
    if "schedule" not in result:
        issues.append("Missing 'schedule' field")
        return issues

    schedule = result["schedule"]

    if len(schedule) != 6:
        found = [d.get("date_label", "?") for d in schedule]
        issues.append(
            f"Expected 6 days (Monday to Saturday), got {len(schedule)}: {found}"
        )

    if len(schedule) > 1:
        tuesday = schedule[1]
        if tuesday.get("slots") is not None:
            issues.append(
                f"Tuesday ({tuesday.get('date_label', '?')}) must have null slots (rest day)"
            )

    MIN_MINS = 9 * 60 + 15   # 09:15
    MAX_MINS = 20 * 60        # 20:00
    total_minutes = 0

    for i, entry in enumerate(schedule):
        label = entry.get("date_label", f"day {i + 1}")
        slots = entry.get("slots")

        if slots is None:
            continue
        if not isinstance(slots, dict):
            issues.append(f"{label}: 'slots' must be an object or null, got {type(slots).__name__}")
            continue

        if slots:
            missing = [k for k in ("morning", "afternoon") if k not in slots]
            if missing:
                issues.append(
                    f"{label}: missing slot(s) {missing} — both 'morning' and 'afternoon' are required"
                )

        for slot_name, slot in slots.items():
            if slot_name not in ("morning", "afternoon"):
                issues.append(f"{label}: unexpected slot key '{slot_name}' (expected 'morning' or 'afternoon')")
                continue

            start = slot.get("start", "")
            end = slot.get("end", "")
            s_mins = _parse_hhmm(start)
            e_mins = _parse_hhmm(end)

            if s_mins is None:
                issues.append(f"{label} {slot_name}: invalid start time '{start}' (expected hh:mm)")
                continue
            if e_mins is None:
                issues.append(f"{label} {slot_name}: invalid end time '{end}' (expected hh:mm)")
                continue

            if s_mins < MIN_MINS:
                issues.append(f"{label} {slot_name}: start {start} is before 09:15")
            if e_mins > MAX_MINS:
                issues.append(f"{label} {slot_name}: end {end} is after 20:00")
            if e_mins <= s_mins:
                issues.append(f"{label} {slot_name}: end {end} is not after start {start}")
            else:
                total_minutes += e_mins - s_mins

    if "weekly_hours" in result and not issues:
        expected = _parse_weekly_hours(str(result["weekly_hours"]))
        if expected is not None:
            diff = abs(total_minutes - expected)
            if diff > 30:
                th, tm = divmod(total_minutes, 60)
                eh, em = divmod(expected, 60)
                issues.append(
                    f"Total computed hours ({th}h{tm:02d}) do not match weekly_hours "
                    f"'{result['weekly_hours']}' ({eh}h{em:02d}). Please recheck the slots."
                )

    return issues


def build_reprompt(issues: list[str]) -> str:
    lines = [
        "Look at the planning image again carefully.",
        "",
        "Your previous answer contained the following errors:",
    ]
    for issue in issues:
        lines.append(f"- {issue}")
    lines += [
        "",
        "IMPORTANT INSTRUCTIONS:",
        "- Re-read the image to find the correct values.",
        "- If you cannot read a value clearly, return {\"error\": \"cannot read <field> clearly\"}.",
        "- If you cannot fix all the issues listed above, return {\"error\": \"<description of what you cannot resolve>\"}.",
        "- Do NOT repeat the same wrong values.",
        "- Do NOT guess or invent values you cannot see in the image.",
        "- Return ONLY the corrected JSON. No markdown, no explanation.",
    ]
    return "\n".join(lines)


def ocr_check(image_path: str) -> list[str]:
    """Run OCR on the image and verify required keywords are present before calling the AI."""
    img = cv2.imread(image_path)
    if img is None:
        return ["cannot read image file"]

    # Upscale small images for better OCR accuracy
    h, w = img.shape[:2]
    if w < 1500:
        scale = 1500 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu binarization — critical for OCR on photos of printed documents
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        # PSM 6: uniform block of text — better for tables than PSM 11 (sparse)
        text = pytesseract.image_to_string(binarized, config="--psm 6 --oem 3")
        if not text.strip():
            # Fallback to sparse mode if block mode returns nothing
            text = pytesseract.image_to_string(binarized, config="--psm 11 --oem 3")
    except pytesseract.pytesseract.TesseractNotFoundError:
        print("  [pre-check] tesseract not installed — skipping OCR pre-check", file=sys.stderr)
        return []

    text_upper = text.upper()
    print(f"  [pre-check] OCR full text:\n{text_upper}\n---", file=sys.stderr)

    errors = []

    if "NOM" not in text_upper:
        errors.append("'NOM' column header not found — this does not look like a planning table")

    # Accept partial matches to account for OCR noise (e.g. "TRESALLE" or "STEPHAN")
    tresallet_found = "TRESALLET" in text_upper or "TRESALLE" in text_upper
    stephanie_found = "STEPHANIE" in text_upper or "STEPHAN" in text_upper
    if not tresallet_found and not stephanie_found:
        errors.append("'TRESALLET' or 'STEPHANIE' not found — Stéphanie's row is missing")

    # S<N> with or without word boundary (OCR may attach to adjacent chars)
    if not re.search(r"S\d{1,2}", text_upper):
        errors.append("week number 'S<N>' not found — the week indicator is missing or unreadable")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Stéphanie Tresallet's schedule from a planning image via minicpm-v/Ollama"
    )
    parser.add_argument("image", help="Path to the planning image")
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Manual crop region: x y width height (overrides auto-detect)",
    )
    parser.add_argument(
        "--no-auto-crop",
        action="store_true",
        help="Disable automatic table detection and cropping",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=2048,
        help="Max image width sent to the model (default: 2048, only resizes if larger)",
    )
    parser.add_argument(
        "--debug-dir",
        metavar="DIR",
        help="Directory to save one PNG per processing step",
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
        help=f"Ollama base/generate URL (default: {OLLAMA_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout per turn in seconds (default: 600)",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the model loaded in memory after the last request (useful during debug)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Max re-prompt attempts on validation failure (default: {MAX_RETRIES})",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip the OCR pre-check (useful if tesseract is unavailable or unreliable)",
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.skip_ocr:
        print("[0/3] OCR pre-check skipped (--skip-ocr)", file=sys.stderr)
    else:
        print("[0/3] OCR pre-check...", file=sys.stderr)
        ocr_errors = ocr_check(args.image)
        if ocr_errors:
            for err in ocr_errors:
                print(f"  [pre-check] {err}", file=sys.stderr)
            result = {"error": "image pre-check failed: " + "; ".join(ocr_errors)}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(2)
        print("[pre-check] OK — required keywords found", file=sys.stderr)

    debug_dir: Path | None = None
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Preprocessing image...", file=sys.stderr)
    processed = preprocess_image(
        args.image,
        crop=args.crop,
        auto_crop=not args.no_auto_crop,
        max_width=args.max_width,
        debug_dir=debug_dir,
    )

    print("[2/3] Encoding image...", file=sys.stderr)
    image_b64 = encode_image(processed)

    # Build initial message with the image attached
    messages: list[dict] = [
        {"role": "user", "content": PROMPT, "images": [image_b64]}
    ]

    result: dict = {}
    for attempt in range(1, args.max_retries + 1):
        is_last = attempt == args.max_retries
        print(
            f"[3/3] Querying minicpm-v (attempt {attempt}/{args.max_retries})...",
            file=sys.stderr,
        )

        raw, messages = chat_ollama(
            messages,
            url=args.ollama_url,
            timeout=args.timeout,
            keep_alive=True if not is_last else args.keep_alive,
        )

        try:
            result = parse_json_response(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"[validation] JSON parse error: {exc}", file=sys.stderr)
            if not is_last:
                reprompt = (
                    "Look at the planning image again carefully.\n"
                    f"Your response could not be parsed as valid JSON: {exc}\n"
                    "If you cannot produce a valid JSON, return {\"error\": \"cannot produce valid JSON\"}.\n"
                    "Return ONLY the JSON, no markdown, no explanation."
                )
                messages.append({"role": "user", "content": reprompt, "images": [image_b64]})
            continue

        issues = validate_schedule(result)
        if not issues:
            print("[validation] OK", file=sys.stderr)
            break

        print(f"[validation] {len(issues)} issue(s) found:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)

        if not is_last:
            # Re-attach the image so the model can re-read it instead of
            # relying on context (vision models often drop the image after turn 1)
            messages.append({
                "role": "user",
                "content": build_reprompt(issues),
                "images": [image_b64],
            })
        else:
            print("[validation] Max retries reached, returning last result.", file=sys.stderr)

    if "error" in result:
        print(f"Error from model: {result['error']}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
