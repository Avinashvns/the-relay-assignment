import csv
import sys
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
RESULTS_FILE = PROJECT_ROOT / "results.csv"

sys.path.insert(0, str(SRC_PATH))

from relay.providers.standin import StandInProvider  # noqa: E402


CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]


def run_one_request(
    provider,
    scenario,
    *,
    input_tokens=974,
    output_tokens=319,
    start_gate=None,
):
    if start_gate is not None:
        start_gate.wait(timeout=15)

    started_at = datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    )
    started = perf_counter()

    row = {
        "request_id": str(uuid.uuid4()),
        "scenario": scenario,
        "source": "stand-in",
        "start_time": started_at,
        "first_token_ms": "",
        "total_ms": "",
        "status": "ok",
        "retries": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "retrieval_setting": "",
        "answer_grade": "",
    }

    try:
        for chunk in provider.stream(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ):
            if row["first_token_ms"] == "":
                row["first_token_ms"] = round(
                    (perf_counter() - started) * 1000,
                    1,
                )

            row["tokens_in"] += chunk.get("tokens_in", 0)
            row["tokens_out"] += chunk.get("tokens_out", 0)

    except Exception as error:
        status_code = getattr(error, "status_code", None)

        if status_code == 429:
            row["status"] = "rate_limited"
        elif status_code == 503:
            # The stand-in reports overload as HTTP 503.
            # The assignment's allowed CSV status for this is "error".
            row["status"] = "error"
        elif "timeout" in type(error).__name__.lower():
            row["status"] = "timeout"
        else:
            row["status"] = "error"

    row["total_ms"] = round((perf_counter() - started) * 1000, 1)
    return row


def run_concurrent_requests(
    provider,
    scenario,
    count,
    *,
    input_tokens=974,
    output_tokens=319,
):
    start_gate = Barrier(count)

    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(
                run_one_request,
                provider,
                scenario,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                start_gate=start_gate,
            )
            for _ in range(count)
        ]

        return [future.result() for future in futures]


def save_rows(rows):
    file_exists = RESULTS_FILE.exists()

    if file_exists:
        with RESULTS_FILE.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            existing_header = next(csv.reader(file), [])

        if existing_header != CSV_COLUMNS:
            raise SystemExit(
                "results.csv does not have the expected 12-column header. "
                "Do not append stand-in rows until the header is corrected."
            )

    with RESULTS_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def print_summary(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["scenario"]].append(row)

    print("\nStand-in load-test summary")
    print("-" * 72)

    for scenario, scenario_rows in grouped.items():
        statuses = Counter(row["status"] for row in scenario_rows)
        successful_times = [
            row["total_ms"]
            for row in scenario_rows
            if row["status"] == "ok"
        ]

        mean_time = (
            round(sum(successful_times) / len(successful_times), 1)
            if successful_times
            else "n/a"
        )

        print(
            f"{scenario}: "
            f"requests={len(scenario_rows)}, "
            f"outcomes={dict(statuses)}, "
            f"mean_success_time_ms={mean_time}"
        )


def main():
    all_rows = []

    print("Scenario 1: steady requests")
    steady_provider = StandInProvider()

    for _ in range(4):
        all_rows.append(
            run_one_request(steady_provider, "steady")
        )

    print("Scenario 2: burst of 5 simultaneous requests")
    burst_provider = StandInProvider(
        max_concurrency=5,
        extra_load_delay_ms=150,
    )
    all_rows.extend(
        run_concurrent_requests(
            burst_provider,
            "burst",
            count=5,
        )
    )

    print("Scenario 3: sustained overload across 8 concurrent waves")
    overload_provider = StandInProvider(
        max_concurrency=3,
        extra_load_delay_ms=150,
    )

    for wave in range(1, 9):
        print(f"  Wave {wave}/8")
        all_rows.extend(
            run_concurrent_requests(
                overload_provider,
                "sustained_overload_8_waves",
                count=8,
                input_tokens=1,
                output_tokens=1,
            )
        )

    print("Scenario 4: requests-per-minute limit halved from 30 to 15")
    halved_rpm_provider = StandInProvider(
        requests_per_minute=15,
        ttft_ms=1,
        output_tokens_per_second=100000,
    )

    for _ in range(16):
        all_rows.append(
            run_one_request(
                halved_rpm_provider,
                "rpm_halved",
                input_tokens=1,
                output_tokens=1,
            )
        )

    print("Scenario 5: exhaust a three-request daily budget")
    daily_limit_provider = StandInProvider(
        requests_per_day=3,
        ttft_ms=1,
        output_tokens_per_second=100000,
    )

    for _ in range(4):
        all_rows.append(
            run_one_request(
                daily_limit_provider,
                "daily_budget_exhausted",
                input_tokens=1,
                output_tokens=1,
            )
        )

    save_rows(all_rows)
    print_summary(all_rows)
    print(f"\nStand-in rows appended to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()