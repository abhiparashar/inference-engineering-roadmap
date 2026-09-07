# 6 — Build: A Web Server From Scratch, Then With a Framework

> **Goal:** turn the abstract systems ideas (syscalls, concurrency, async) into muscle memory by building a tiny HTTP server two ways — raw sockets, then FastAPI — and *seeing* exactly what the framework does for you.

This is the Phase 0 build. It's small on purpose. The value isn't the code; it's that afterward you'll never again treat a web framework as magic — which matters because in Phase 3 you'll be *modifying* how a server batches and schedules requests.

---

## Why build a server at all in an "inference" roadmap?

Because **an inference service is a web server that happens to run a model.** Every serving framework (vLLM, TGI, Triton) is, at its heart: accept HTTP requests → parse them → do model work → stream results back. If you understand the plain version, the fancy versions are just that plus batching and GPUs. So we start at the bottom: what is HTTP, really, and what does a framework hide?

---

## Part A — The raw-socket server (no framework)

### What HTTP actually is

HTTP is just **text sent over a TCP connection** in an agreed format. A request looks like this (literally these bytes):

```
GET /health HTTP/1.1\r\n
Host: localhost\r\n
\r\n
```

And a response looks like this:

```
HTTP/1.1 200 OK\r\n
Content-Type: application/json\r\n
Content-Length: 15\r\n
\r\n
{"status":"ok"}
```

That's the whole secret. `\r\n` (carriage-return + newline) separates lines; a blank line separates headers from the body. A "web server" is a program that reads those request bytes off a socket and writes those response bytes back.

### The code

A **socket** is your program's endpoint for network communication — the thing you `recv` from and `send` to (remember syscalls from [lesson 1](01-how-a-computer-runs-your-code.md)?). Here's a complete server, standard library only:

```python
# raw_server.py — an HTTP server with zero frameworks
import socket

HOST, PORT = "127.0.0.1", 8000

def handle(request_text: str) -> bytes:
    # Parse the request line: e.g. "GET /health HTTP/1.1"
    first_line = request_text.split("\r\n", 1)[0]
    parts = first_line.split(" ")
    method = parts[0] if len(parts) > 0 else ""
    path   = parts[1] if len(parts) > 1 else "/"

    if method == "GET" and path == "/health":
        body = b'{"status":"ok"}'
        status = "200 OK"
    else:
        body = b'{"error":"not found"}'
        status = "404 Not Found"

    headers = (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode()
    return headers + body

def main():
    # AF_INET = IPv4, SOCK_STREAM = TCP
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # reuse port on restart
    server.bind((HOST, PORT))     # claim the address/port  (syscall)
    server.listen(128)            # start accepting; 128 = backlog queue
    print(f"listening on http://{HOST}:{PORT}")

    while True:
        conn, addr = server.accept()          # WAIT for a client  (blocking syscall)
        with conn:
            request_bytes = conn.recv(65536)   # read request bytes (blocking syscall)
            if not request_bytes:
                continue
            response = handle(request_bytes.decode("utf-8", "replace"))
            conn.sendall(response)             # write response bytes (syscall)

if __name__ == "__main__":
    main()
```

Run it (`python raw_server.py`) and test it:

```bash
curl -i http://127.0.0.1:8000/health     # 200 OK, {"status":"ok"}
curl -i http://127.0.0.1:8000/nope       # 404 Not Found
```

### What you just did (and its flaws — this is the important part)

You wrote, by hand: bound a socket, accepted a connection (`accept` — a **blocking** syscall that *waits*), read bytes (`recv` — also blocking), parsed HTTP text yourself, and formatted a response by hand. Notice the flaws, because they're exactly what frameworks fix:

1. **One request at a time.** The loop `accept → recv → send` handles a single client fully before touching the next. If two clients connect at once, the second *waits*. This is the "blocking" server — pure sequential, no concurrency. (Connect the dots to [lesson 3](03-processes-threads-concurrency.md): all that `accept`/`recv` waiting is I/O-bound time where the CPU sits idle.)
2. **Fragile parsing.** Real HTTP has methods, query strings, headers, chunked bodies, keep-alive, malformed input… you'd hand-write all of it.
3. **No routing, validation, JSON handling, error pages, concurrency, or graceful shutdown.**

That list *is* the job description of a web framework.

---

## Part B — The same thing with FastAPI

Now the framework version:

```python
# app.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}       # framework serializes to JSON + sets headers
```

Run it with an ASGI server (`uvicorn`):

```bash
pip install fastapi uvicorn
uvicorn app:app --host 127.0.0.1 --port 8000
curl -i http://127.0.0.1:8000/health
```

Same behavior, a fraction of the code. **So what did FastAPI + uvicorn do that you had to do by hand?**

| You did by hand (raw) | FastAPI / uvicorn does for you |
|---|---|
| `socket` / `bind` / `listen` / `accept` | uvicorn owns the socket lifecycle |
| Blocking, one-at-a-time loop | Runs on an **async event loop** → juggles many connections concurrently ([lesson 3](03-processes-threads-concurrency.md)) |
| Parse the raw HTTP text yourself | Full, correct HTTP parsing (methods, headers, query, body, keep-alive) |
| `if path == "/health"` routing by hand | `@app.get("/health")` decorator routing |
| Hand-format headers + `Content-Length` | Automatic; also JSON serialization of your return value |
| No validation | Pydantic request/response validation and auto-docs |
| Crash on bad input | Structured error responses |

The single biggest thing: **uvicorn runs your handlers on an async event loop**, so it handles thousands of concurrent connections on one thread by overlapping their I/O waits — the concurrency you'd otherwise have to build with threads/processes yourself. This is the exact machinery every inference server sits on top of.

### Prove the concurrency difference

Add a slow endpoint to *both* servers and hit it with 5 simultaneous requests:

```python
# FastAPI version — async, so waits overlap
import asyncio
@app.get("/slow")
async def slow():
    await asyncio.sleep(1.0)      # pretend we're waiting on something
    return {"done": True}
```

```bash
# fire 5 at once; total ≈ 1s on FastAPI (overlapped), ≈ 5s on the raw blocking server
time (for i in $(seq 5); do curl -s http://127.0.0.1:8000/slow & done; wait)
```

The raw server serializes the five 1-second waits into ~5s; FastAPI overlaps them into ~1s. That gap *is* concurrency, and it's why the async lesson mattered.

> **Sharp edge that foreshadows Phase 3:** make the FastAPI endpoint `def slow()` (not `async`) and put a **CPU-busy loop** inside instead of `asyncio.sleep`. Now it blocks the event loop and concurrency collapses — exactly the "heavy compute inside the loop freezes everything" warning from lesson 3. This is *precisely* why real inference servers push model compute off the event loop.

---

## Deliverable for this lesson

Commit both servers and a short note. Suggested location: `projects/phase0-raw-vs-framework-server/` in this repo (see [`projects/`](../../projects/README.md)).

- `raw_server.py` — the from-scratch version
- `app.py` — the FastAPI version
- `NOTES.md` — 5-10 bullets answering: *what did the framework do for me?*, plus your measured numbers from the 5-concurrent-requests test (raw ≈ 5s, FastAPI ≈ 1s or similar).

That NOTES.md is your understanding made visible. It rolls into the Phase 0 exit artifact in [lesson 7](07-exercises-and-artifacts.md).

---

## Key takeaways

- HTTP is just formatted **text over a TCP connection**; a server reads request bytes and writes response bytes.
- Writing it raw exposes the real syscalls (`bind`/`listen`/`accept`/`recv`/`send`) and their **blocking, one-at-a-time** nature.
- A framework (FastAPI) + ASGI server (uvicorn) gives you correct HTTP parsing, routing, validation, JSON — and crucially an **async event loop** for concurrency.
- Heavy compute inside the event loop kills concurrency — the reason inference servers isolate model work. Hold this thought straight into Phase 3.

**Next:** [Exercises & exit artifact →](07-exercises-and-artifacts.md)
