## Measured results

Every value below retains all four reports per arm. Aggregate cells are median (minimum–maximum) across report values; these ranges are not confidence intervals.

| Context | Control A novel decode tok/s | PR B novel decode tok/s | Median change | A prefill tok/s | B prefill tok/s |
|---|---:|---:|---:|---:|---:|
| 2K | 49.7 (48.3–51.0) | 47.2 (42.7–54.0) | -5.0% | 669.0 (647.0–737.0) | 666.5 (628.0–710.0) |
| 8K | 43.8 (40.3–46.5) | 40.7 (39.4–45.2) | -7.0% | 718.5 (653.0–730.0) | 699.0 (617.0–746.0) |
| 16K | 43.5 (40.9–44.3) | 41.3 (38.8–42.9) | -4.9% | 696.0 (666.0–740.0) | 644.5 (635.0–668.0) |
| 32K | 36.5 (34.3–37.8) | 31.2 (29.7–40.2) | -14.7% | 631.5 (613.0–707.0) | 634.0 (629.0–646.0) |

| Short-prompt metric | Control A | PR B |
|---|---:|---:|
| decode | 42.3 (39.4–42.7) | 38.1 (36.8–39.6) |
| novel_spec | 20.0 (19.6–21.3) | 19.6 (19.1–20.0) |
| predictable_spec | 84.4 (82.2–88.0) | 80.9 (77.0–83.3) |
| prefill | 735.9 (707.3–750.8) | 631.8 (596.6–677.1) |
| ttft_ms | 371.5 (364.0–380.0) | 407.5 (394.0–415.0) |

## Individual reports

| Run | Novel spec tok/s | 2K novel tok/s | 8K | 16K | 32K | Load drift |
|---|---:|---:|---:|---:|---:|---|
| [01-A](01-A.llmprobe.json) | 19.9 | 48.3 | 42.5 | 43.4 | 37.8 | -5.6% (steady) |
| [02-B](02-B.llmprobe.json) | 20 | 46 | 41.8 | 42.9 | 30.1 | -2.8% (steady) |
| [03-B](03-B.llmprobe.json) | 19.1 | 48.4 | 39.6 | 38.8 | 32.2 | +4.5% (steady) |
| [04-A](04-A.llmprobe.json) | 19.6 | 51 | 46.5 | 44.3 | 35.7 | -5.4% (steady) |
| [05-A](05-A.llmprobe.json) | 20.1 | 50.2 | 45 | 40.9 | 34.3 | -12.4% (degraded) |
| [06-B](06-B.llmprobe.json) | 19.9 | 54 | 39.4 | 42.8 | 29.7 | -1.8% (steady) |
| [07-B](07-B.llmprobe.json) | 19.3 | 42.7 | 45.2 | 39.8 | 40.2 | +8.7% (steady) |
| [08-A](08-A.llmprobe.json) | 21.3 | 49.2 | 40.3 | 43.5 | 37.3 | -9.8% (steady) |

## Engagement

| Run | Width-9 split lines | Depth-8 MTP trace lines |
|---|---:|---:|
| 01-A | 1 | 27 |
| 02-B | 0 | 27 |
| 03-B | 0 | 27 |
| 04-A | 1 | 27 |
| 05-A | 1 | 27 |
| 06-B | 0 | 27 |
| 07-B | 0 | 26 |
| 08-A | 1 | 27 |
