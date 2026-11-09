## 1. What I built and why



I built a Python research-paper assistant for a fixed library of three papers. The user selects one of three retrieval settings and asks a question. BM25 ranks the prepared paper chunks, and the selected passages are sent to Groq GPT-OSS-120B. The model streams a cited answer or abstains when the retrieved evidence is insufficient. The program checks source markers, asks me to grade the answer, and records request metrics and the grade in `results.csv`.



The pipeline branches when no relevant passages are retrieved, and it processes the model response chunk by chunk. My main design decision was to prefer an evidence-backed answer or an explicit refusal over guessing. This can leave some questions unanswered, but it reduces unsupported claims.



## 2. The real service and my stand-in



### Table 1: Limits found in the real service



| Limit | Value found | How found | Confidence |
|---|---:|---|---|
| Requests per minute | 30 | Groq account Limits page | High |
| Tokens per minute | 8,000 | Groq Limits page and response header | High |
| Requests per day | 1,000 | Groq Limits page; the `x-ratelimit-limit-requests` header reports RPD | High |
| Tokens per day | 200,000 | Groq account Limits page | High |
| Requests in flight at once | Not measured | Concurrency was not tested against the real service | Low |
| What it returns when a limit is hit | HTTP 429 Too Many Requests | Groq documentation; no 429 was observed in my requests | Medium |
| Time until it recovers | Not measured | No real 429 response was observed | Low |

Groq documents that rate-limited requests return HTTP 429 and that `Retry-After` is provided on a 429 response. I did not trigger a real limit, so I have not measured the recovery time. Source: [Groq rate-limit documentation](https://console.groq.com/docs/rate-limits).



### Stand-in behavior and calibration



The stand-in is a local Python provider. Its baseline time to first token is 806 ms, and its simulated output rate is 775 tokens per second. These values were calibrated from one real Groq request with a time to first token of 806.3 ms, a total time of 1,218.2 ms, and 319 output tokens. In the stand-in's steady scenario, four requests had a mean total time of 1,199.2 ms. This is an initial calibration, not a reliable latency distribution, because it is based on only one real reference request.



The stand-in can stream placeholder chunks, enforce simulated per-minute and per-day limits, fail when concurrency is exceeded, and add delay under load. Its biggest difference from Groq is that it runs in-process and returns placeholder chunks rather than making a network request to a real model. The current load-test script tests the stand-in provider directly; it does not yet load-test the complete retrieval-and-answer pipeline.





## 3. Results



### Table 2a: Latency by scenario

| Scenario | Service | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---:|---:|---:|---:|---:|
| steady | stand-in | 8 | not measured | not measured | 806.5 / 806.6 / 806.7 | 1198.8 / 1201.1 / 1201.1 |
| burst | stand-in | 10 | not measured | not measured | 1106.3 / 1406.8 / 1406.9 | 1500.7 / 1801.8 / 1802.1 |
| sustained_overload | stand-in | 8 | not measured | not measured | 956.4 / 1091.2 / 1103.2 | 956.4 / 1091.2 / 1103.2 |
| sustained_overload_8_waves | stand-in | 64 | not measured | not measured | 956.4 / 1106.8 / 1107.0 | 956.4 / 1106.9 / 1107.0 |
| rpm_halved | stand-in | 32 | not measured | not measured | 1.6 / 1.9 / 1.9 | 1.6 / 1.9 / 1.9 |
| daily_budget_exhausted | stand-in | 8 | not measured | not measured | 1.6 / 1.9 / 1.9 | 1.6 / 1.9 / 1.9 |
| interactive_demo | real | 3 | not measured | not measured | 931.0 / 1121.7 / 1138.7 | 1251.3 / 1526.2 / 1550.6 |
| smoke_test | real | 24 | not measured | not measured | 1350.3 / 1648.6 / 1750.5 | 1582.0 / 2067.8 / 2206.0 |

### Table 2b: Outcomes by scenario

| Scenario | Succeeded (%) | Rate-limited (%) | Timed out / errors / given up (%) | Retries per request | Peak queue length | Tokens per successful answer | Cost per 1,000 answers |
|---|---:|---:|---:|---:|---|---:|---|
| steady | 100.0 | 0.0 | 0.0 | 0.0 | not measured | 1293.0 | not measured |
| burst | 100.0 | 0.0 | 0.0 | 0.0 | not measured | 1293.0 | not measured |
| sustained_overload | 37.5 | 0.0 | 62.5 | 0.0 | not measured | 2.0 | not measured |
| sustained_overload_8_waves | 37.5 | 0.0 | 62.5 | 0.0 | not measured | 2.0 | not measured |
| rpm_halved | 93.8 | 6.2 | 0.0 | 0.0 | not measured | 2.0 | not measured |
| daily_budget_exhausted | 75.0 | 25.0 | 0.0 | 0.0 | not measured | 2.0 | not measured |
| interactive_demo | 100.0 | 0.0 | 0.0 | 0.0 | not measured | 2205.0 | not measured |
| smoke_test | 91.7 | 0.0 | 8.3 | 0.0 | not measured | 2206.1 | not measured |

The raw summary combines separate executions that share the same scenario name, so its timestamp span includes idle time between runs. Duration and offered rate are therefore marked “not measured.” Queue length and monetary cost were not recorded. The real smoke-test errors and stand-in overload errors are included in the timed-out/errors/given-up column.


### Table 3: Retrieval settings



| Setting (chunk size; overlap; pieces fetched; re-rank) | Questions asked | Correct and cited | Wrong | Correctly refused | Mean tokens per answer (input + output) | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| small_k3 (180 words; 30-word overlap; 3; no) | 5 | 20.0% | 60.0% | 20.0% | 1343.0 | 1363.1 / 1650.8 | 1551.9 / 2242.2 |
| medium_k4 (320 words; 50-word overlap; 4; no) | 5 | 80.0% | 20.0% | 0.0% | 2362.6 | 1428.4 / 1777.0 | 1855.4 / 2069.6 |
| large_k5_rerank (500 words; 80-word overlap; 5; yes) | 5 | 20.0% | 80.0% | 0.0% | 3812.4 | 1382.4 / 1468.9 | 1785.2 / 1963.2 |

“Wrong” includes incorrect answers and incorrect refusals. These are 15 graded requests, with five questions per setting. Each setting received different questions, so the results are indicative and do not isolate chunk size or reranking as the cause of the quality differences.



## 4. Where the time goes



The slowest successful real request in the current results was request `d8c6ab46-891b-4cd9-8fd0-efdd1956c4cc`, using `small_k3` in the `smoke_test` scenario. Its time to first token was 1,606.3 ms and its total pipeline time was 2,242.2 ms. The remaining 635.9 ms elapsed after the first token while the response stream was being received. It used zero retries.



The timer starts before retrieval, and the current CSV does not record retrieval duration and Groq request duration separately. Therefore, the 1,337.5 ms before the first token combines retrieval, request setup, network/model wait, and time to the first streamed content. I cannot identify how much was waiting versus model work from these measurements.



## 5. When a limit is hit

I found account limits of 30 requests per minute, 8,000 tokens per minute, 1,000 requests per day, and 200,000 tokens per day. I did not hit a real Groq limit. When the client receives HTTP 429, it records `rate_limited` in `results.csv`; `max_retries=0`, so the request is not retried or queued. The user sees that the request failed.

In the stand-in tests, halving the per-minute request budget produced 2 rate-limited requests out of 32 (6.2%). The daily-budget tests produced 2 rate-limited requests out of 8 (25.0%). During the 8-wave overload test, 40 of 64 requests failed (62.5%) because the concurrency cap was exceeded. Excess requests are dropped instead of delayed; the system does not return a deliberately weaker answer. A user can manually choose `small_k3` to retrieve fewer passages near a limit, but automatic switching was not implemented or measured.

## Retrieval: quality versus cost
Across 15 graded real requests, with five questions per setting, `medium_k4` had the highest correct-and-cited rate at 80.0%. `small_k3` used the fewest mean tokens per answer (1,343.0), but had 20.0% correct-and-cited, 60.0% wrong, and 20.0% correctly refused. `large_k5_rerank` had 20.0% correct-and-cited and the highest mean token use (3,812.4). Since each setting received different questions and the sample was small, this comparison is indicative rather than controlled.

I would choose `medium_k4` as the default because it had the strongest observed answer quality. Near a limit, I would manually switch to `small_k3` to retrieve fewer passages and reduce token use, while accepting less evidence in the context. Automatic switching was not implemented or measured.

## How this library will mislead a naive retriever

First, keyword overlap can make an out-of-scope question look answerable. For “What is an LLM?”, the papers returned no sufficient definition, so the assistant correctly abstained. A naive system might still present a loosely related Transformer passage as a definition.

Second, related passages may not support every claim in a generated answer. For “What role does the Mimi codec play in Moshi’s real-time speech-text system?”, the answer included a teacher-model claim that its cited passages did not support. The relevant topic matched, but the evidence for that specific claim needed checking.

Third, a paper-supported question can still fail in the retrieval-and-citation pipeline. For “How does Moshi model the user's and assistant's speech streams to support full-duplex dialogue?”, the paper contains the answer, but the system returned a citation-verification fallback. This run does not show whether retrieval or citation validation caused the failure.

## 6. Retry policy

The Groq client uses `max_retries=0`, so it does not automatically retry failed requests. In the stand-in `rpm_halved` runs, 2 of 32 requests were rate-limited. An immediate retry before the limit resets would likely receive the same rate-limit response, adding another attempt and more waiting without producing an answer. This is a reasoned failure case based on the observed stand-in limit behavior; I did not run a retry-enabled experiment, so the extra latency was not directly measured.

If retries are added later, I would retry at most once, only when a usable `Retry-After` delay is available, and record every attempt.



## 7. What the person sees

If the daily budget is exhausted at 3pm, the current CLI receives a rate-limit error, prints a technical failure message and request ID, and records the request as `rate_limited` in `results.csv`. It does not queue the question, retry it, or return a weaker answer. I did not exhaust the real Groq daily budget; the stand-in test simulated a three-request budget, where the fourth request was rate-limited.

A clearer message for the waiting person would be: “Today's request limit has been reached, so your question was not answered. Please try again after the limit resets. Your request ID is [request ID].”



## 8. What I'm unsure about, what I left out, what I learned

My least certain assumption is that one real Groq request was enough to calibrate the stand-in's latency. If that request was unusually fast or slow, the stand-in's latency results will not represent typical Groq behavior.

I deliberately left out real-service overload testing to stay within my free-tier limits, and automatic retry/queueing because I did not measure real 429 recovery behavior. As a result, the stand-in tests show simulated limits, not how Groq behaves under sustained overload.

One thing I learned is that time to first token and total time answer different questions. The first includes retrieval and the wait for the model's first streamed content; total time also includes receiving the rest of the answer. My current logs do not separate retrieval time from provider time.



## Your feedback on Vaani

I visited [Vaani's website](https://www.vaani.dev/). What stood out was its focus on natural turn-taking, multilingual voice agents, and the Teacher AI workflow that asks clarifying questions and lets the builder review changes before approval. The site advertises low latency, including a p90 figure, but I could not tell from the page how that figure was measured. I would add a short end-to-end demo and explain the latency measurement conditions. 



## How I worked with the assistant

I used an AI coding assistant during development. Three prompts that mattered were:

1. “maine project update karke run kiya ab ek bar pura code check karao ki kaise kam kar rha hai abhi tak code” — I wanted to understand how the existing pipeline worked before changing it.
2. “tum hi code sahi karke do” — I wanted help correcting the stand-in load-test script while keeping the required CSV schema.
3. “check the answer and its citation against me kya dalu” — I needed to verify whether an answer and its citations deserved a correct, wrong, or refusal grade.

The assistant initially told me to grade the out-of-scope “what is llm” refusal as `correct`. I checked the result and changed the grade to `refused_correctly`. I also overruled the assistant’s batch-run suggestion and chose to run the remaining questions one at a time so I could inspect each answer and citation before grading it.

The measurements, model responses, citation checks, grades, and decision to run questions individually came from my own project work. The code and report were AI-assisted.

The prompts above are excerpts. Attach the unedited conversation history separately if the assignment requires the full conversation and you can export it.