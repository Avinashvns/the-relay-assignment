import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "results.csv"


def as_number(value):
    try:
        if value is None or value.strip() == "":
            return None
        return float(value)
    except (AttributeError, ValueError):
        return None


def percentile(values, fraction):
    """Calculate a linearly interpolated percentile."""
    values = sorted(values)

    if not values:
        return None

    position = (len(values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return values[lower_index]

    weight = position - lower_index
    return (
        values[lower_index] * (1 - weight)
        + values[upper_index] * weight
    )


def show(value):
    if value is None:
        return ""
    return f"{value:.1f}"


def main():
    if not RESULTS_FILE.exists():
        raise SystemExit(f"Could not find: {RESULTS_FILE}")

    groups = defaultdict(list)

    with RESULTS_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "source",
            "scenario",
            "status",
            "first_token_ms",
            "total_ms",
            "retries",
            "tokens_in",
            "tokens_out",
        }

        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"results.csv is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            groups[(row["source"], row["scenario"])].append(row)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow([
        "source",
        "scenario",
        "requests",
        "succeeded_pct",
        "rate_limited_pct",
        "timeout_or_error_pct",
        "mean_retries_per_request",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "total_p50_ms",
        "total_p95_ms",
        "total_p99_ms",
        "mean_tokens_per_successful_answer",
    ])

    for (source, scenario), rows in sorted(groups.items()):
        total_requests = len(rows)
        successful = [row for row in rows if row["status"] == "ok"]

        def percentage(status):
            count = sum(row["status"] == status for row in rows)
            return 100 * count / total_requests if total_requests else 0

        failed_pct = 100 * sum(
            row["status"] in {"timeout", "error"}
            for row in rows
        ) / total_requests

        ttft_values = [
            value
            for row in successful
            if (value := as_number(row["first_token_ms"])) is not None
        ]
        total_values = [
            value
            for row in successful
            if (value := as_number(row["total_ms"])) is not None
        ]
        retry_values = [
            value
            for row in rows
            if (value := as_number(row["retries"])) is not None
        ]
        token_totals = [
            input_tokens + output_tokens
            for row in successful
            if (input_tokens := as_number(row["tokens_in"])) is not None
            and (output_tokens := as_number(row["tokens_out"])) is not None
        ]

        mean_retries = (
            sum(retry_values) / len(retry_values)
            if retry_values
            else None
        )
        mean_tokens = (
            sum(token_totals) / len(token_totals)
            if token_totals
            else None
        )

        writer.writerow([
            source,
            scenario,
            total_requests,
            show(percentage("ok")),
            show(percentage("rate_limited")),
            show(failed_pct),
            show(mean_retries),
            show(percentile(ttft_values, 0.50)),
            show(percentile(ttft_values, 0.95)),
            show(percentile(ttft_values, 0.99)),
            show(percentile(total_values, 0.50)),
            show(percentile(total_values, 0.95)),
            show(percentile(total_values, 0.99)),
            show(mean_tokens),
        ])


if __name__ == "__main__":
    main()