import argparse
import json
from typing import Dict, Generator, Tuple

import requests


def parse_sse_lines(
    lines: Generator[str, None, None],
) -> Generator[Tuple[str, Dict], None, None]:
    event = "message"
    data_lines = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = {"raw": payload}
                yield event, data
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="React-style SSE state parser for /rag/query/stream"
    )
    parser.add_argument("--url", default="http://localhost:8000/rag/query/stream")
    parser.add_argument("--user-id", default="react_test_user")
    parser.add_argument("--question", default="船舶电气设备故障如何快速排查？")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--raw-events",
        action="store_true",
        help="print every SSE event as it arrives",
    )
    parser.add_argument(
        "--final-json-only",
        action="store_true",
        help="only print final assembled JSON state",
    )
    args = parser.parse_args()

    payload = {
        "user_id": args.user_id,
        "question": args.question,
        "top_k": args.top_k,
        "use_kg": True,
        "use_history": True,
    }

    with requests.post(
        args.url,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        json=payload,
        stream=True,
        timeout=600,
    ) as resp:
        resp.raise_for_status()
        state = {
            "answer": "",
            "citations": [],
            "kgTriplets": [],
            "meta": None,
            "done": None,
            "error": None,
            "loading": True,
        }

        if not args.final_json_only:
            print(f"Connected: {resp.status_code} {args.url}")

        for event, data in parse_sse_lines(resp.iter_lines(decode_unicode=True)):
            if args.raw_events and not args.final_json_only:
                print(f"\n[event={event}] {json.dumps(data, ensure_ascii=False)}")

            if event == "meta":
                state["meta"] = data
            elif event == "token":
                token = str(data.get("text", ""))
                if token:
                    state["answer"] += token
                    if not args.final_json_only:
                        print(token, end="", flush=True)
            elif event == "references":
                state["citations"] = data.get("citations", []) or []
            elif event == "kg":
                state["kgTriplets"] = data.get("triplets", []) or []
            elif event == "error":
                state["error"] = data
                state["loading"] = False
                break
            elif event == "done":
                state["done"] = data
                state["loading"] = False
                break

        if state["loading"]:
            state["loading"] = False

        if not args.final_json_only:
            print("\n")
            print("----- Final React-like State JSON -----")
        print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
