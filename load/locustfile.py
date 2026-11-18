"""Load test for the Bento serving surface.

    uv run --group load locust -f load/locustfile.py --host http://127.0.0.1:3000 \
        --users 50 --spawn-rate 5 --run-time 3m --headless \
        --html reports/locust/report.html --csv reports/locust/run

Three things make this a measurement rather than a demonstration:

**The questions are the evaluation set**, read from eval/questions.jsonl --
twenty real bilingual questions about the Egyptian Civil Code. Hammering one
string would let every request after the first hit warm caches at every level
and would understate latency; worse, a single question exercises one
neighbourhood of the index, while these twenty spread across it.

**Streaming is measured by time to first byte, separately from total time.**
Locust's own timing stops when the response completes, which for a streamed
answer is the *last* token -- the number streaming exists to improve is the
first one. ``/ask_stream`` therefore reports two metrics: ``ttfb`` and
``stream``. Reporting only the total would make streaming look like a
regression against ``/ask``.

**The mix is weighted toward /ask.** It is the endpoint a client calls when it
needs the citations, so it is the one whose p95 belongs in the report.
"""

import json
import pathlib
import random
import time

from locust import HttpUser, between, events, task

_QUESTIONS_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"
)

# A short read-time pause. Zero think-time would make 50 users a closed-loop
# hammer rather than 50 people using a service, and would measure the
# threadpool's saturation point instead of latency at a realistic arrival rate.
MIN_WAIT, MAX_WAIT = 0.5, 2.0


def load_questions() -> list[str]:
    """The evaluation questions, or a small fallback if the file is missing.

    The fallback keeps the load test runnable in a fresh clone that has not
    built the corpus yet; it is deliberately bilingual for the same reason the
    real set is.
    """
    if not _QUESTIONS_PATH.is_file():
        return [
            "What is the general period of prescription for obligations?",
            "ما هي المدة العامة لتقادم الالتزامات في القانون المدني؟",
        ]
    questions = []
    for line in _QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(json.loads(line)["question"])
    return questions


QUESTIONS = load_questions()


@events.test_start.add_listener
def _announce(environment, **_kwargs) -> None:
    print(
        f"load test starting against {environment.host} with {len(QUESTIONS)} questions"
    )


class RagUser(HttpUser):
    """One simulated client of the question-answering service."""

    wait_time = between(MIN_WAIT, MAX_WAIT)

    @task(8)
    def ask(self) -> None:
        """The buffered endpoint: the p95 the report quotes.

        ``catch_response`` so a 200 carrying an ungrounded answer is still
        recorded as a failure. A load test that only checks the status code
        would report a service that had lost its index as perfectly healthy at
        0 ms, which is the most misleading green a report can show.
        """
        question = random.choice(QUESTIONS)
        with self.client.post(
            "/ask",
            json={"question": question},
            name="POST /ask",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                body = response.json()
            except ValueError:
                response.failure("response was not JSON")
                return
            if not body.get("sources"):
                response.failure("answer carried no citations")
                return
            response.success()

    @task(2)
    def ask_stream(self) -> None:
        """The streaming endpoint, reported as two metrics.

        ``ttfb`` is time to the first byte of the answer, which is what a
        reader actually waits for. ``stream`` is the full duration, which is
        what Locust would have measured on its own. Both are fired manually
        because ``stream=True`` means the context manager's own timing stops at
        the headers, not at the body.
        """
        question = random.choice(QUESTIONS)
        start = time.perf_counter()
        first_byte_ms: float | None = None
        total_bytes = 0
        error: Exception | None = None

        try:
            with self.client.post(
                "/ask_stream",
                json={"question": question},
                name="POST /ask_stream",
                stream=True,
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}")
                    return
                for chunk in response.iter_content(chunk_size=None):
                    if not chunk:
                        continue
                    if first_byte_ms is None:
                        first_byte_ms = (time.perf_counter() - start) * 1000
                    total_bytes += len(chunk)
                response.success()
        except Exception as exc:
            # Recorded against both metrics below rather than swallowed: a
            # connection reset mid-stream is exactly the failure a load test
            # exists to surface, and re-raising here would kill the user.
            error = exc

        total_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="STREAM",
            name="POST /ask_stream [ttfb]",
            response_time=first_byte_ms if first_byte_ms is not None else total_ms,
            response_length=total_bytes,
            exception=error if first_byte_ms is None else None,
        )
        events.request.fire(
            request_type="STREAM",
            name="POST /ask_stream [total]",
            response_time=total_ms,
            response_length=total_bytes,
            exception=error,
        )

    @task(1)
    def status(self) -> None:
        """Cheap, and the control in the experiment.

        It touches no model, so its latency is the service's own overhead. If
        ``/status`` p95 climbs alongside ``/ask`` p95, the queue is the
        bottleneck rather than inference -- a distinction no amount of staring
        at the /ask number alone can settle.
        """
        self.client.post("/status", json={}, name="POST /status")
