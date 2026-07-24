import math
import os
import time
import math
import os
import signal
import threading
import time

from flask import Flask, jsonify

app = Flask(__name__)
allocated_memory = []

LEAK_KB_PER_REQUEST = int(
    os.getenv("LEAK_KB_PER_REQUEST", "0")
)
CPU_WORK_MS = int(os.getenv("CPU_WORK_MS", "10"))
CRASH_AFTER_SECONDS = int(
    os.getenv("CRASH_AFTER_SECONDS", "0")
)


def terminate_container():
    time.sleep(CRASH_AFTER_SECONDS)
    os.kill(1, signal.SIGTERM)


if CRASH_AFTER_SECONDS > 0:
    threading.Thread(
        target=terminate_container,
        daemon=True,
    ).start()


@app.get("/health")
def health():
    return jsonify(status="healthy"), 200


@app.get("/work")
def work():
    started = time.perf_counter()
    deadline = started + CPU_WORK_MS / 1000

    result = 0.0
    while time.perf_counter() < deadline:
        result += math.sqrt(12345.6789)

    if LEAK_KB_PER_REQUEST > 0:
        allocated_memory.append(
            bytearray(LEAK_KB_PER_REQUEST * 1024)
        )

    duration_ms = (
        time.perf_counter() - started
    ) * 1000

    return jsonify(
        status="completed",
        processing_time_ms=round(duration_ms, 2),
        leaked_kb=LEAK_KB_PER_REQUEST,
        allocation_count=len(allocated_memory),
        result=result,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        threaded=True,

    )