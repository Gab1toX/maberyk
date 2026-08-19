"""Standalone script: read questions from a UTF-8 text file and POST them
to the running web_mind.py server's /queue/enqueue endpoint.

Exists because PowerShell's Invoke-RestMethod encodes request bodies as
Latin-1 by default, silently mangling any question containing ñ or an
accented vowel before it ever reaches the server. This script reads the
file as UTF-8 and sends an explicitly UTF-8-encoded body, sidestepping
Invoke-RestMethod entirely.

File format: one question per line. Blank lines and lines starting with
'#' are ignored.

Usage:
    python -m scripts.enqueue preguntas.txt
    python -m scripts.enqueue preguntas.txt --host localhost --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000


def read_questions(path: Path) -> list[str]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        questions.append(stripped)
    return questions


def enqueue(questions: list[str], host: str, port: int) -> dict:
    body = json.dumps({"questions": questions}).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host}:{port}/queue/enqueue",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="UTF-8 text file with one question per line.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"web_mind.py host (default: {DEFAULT_HOST}).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"web_mind.py port (default: {DEFAULT_PORT}).")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    questions = read_questions(args.file)
    if not questions:
        print(f"error: no questions found in {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        result = enqueue(questions, args.host, args.port)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        print(f"error: server rejected the request: HTTP {exc.code}: {detail}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"error: could not reach {args.host}:{args.port}: {exc.reason}", file=sys.stderr)
        sys.exit(1)

    print(f"Read {len(questions)} question(s) from {args.file}.")
    print(f"Server added {result.get('added', 0)} to the backlog.")


if __name__ == "__main__":
    main()
