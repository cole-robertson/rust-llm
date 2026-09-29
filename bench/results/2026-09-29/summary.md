## Environment

- **host**: framework
- **cpu**: AMD Ryzen AI 9 HX 370 w/ Radeon 890M
- **kernel**: 7.2.3-arch1-3
- **mem**: 93 GiB
- **rustc**: rustc 1.98.1 (48a229cea 2026-09-01) (Arch Linux rust 1:1.98.1-1)
- **ruby**: ruby 3.4.10 (2026-06-30 revision 2b0b7728dc) +PRISM [x86_64-linux]
- **gems**: * async (2.46.0)
  * faraday (2.14.4)
  * json (3.0.2)
  * ruby_llm (2.0.0)
- **loadavg**: ['4.22', '3.07', '1.99']

## Latency (ms unless noted)

| Case | Metric | rust | ruby+yjit | ruby |
|---|---|---:|---:|---:|
| first | process start -> first answer | 10.1 (p90 10.7) | 251.2 (p90 256.5) | 214.6 (p90 223.7) |
| overhead | per-request overhead p50 | 0.080 (0.078-0.081) | 0.246 (0.242-0.258) | 0.410 (0.403-0.414) |
| overhead | per-request overhead p99 | 0.119 (0.112-0.131) | 0.742 (0.716-0.912) | 0.866 (0.845-0.934) |
| stream | streaming, us per chunk p50 | 2.72 (2.70-2.78) | 20.02 (19.97-21.64) | 32.82 (32.77-33.23) |
| tools | 3-round tool loop p50 | 0.383 (0.353-0.386) | 1.046 (1.033-1.086) | 1.684 (1.643-2.056) |
| tools | 3-round tool loop p99 | 0.530 (0.424-0.546) | 2.848 (2.551-4.156) | 3.778 (2.574-3.926) |
| render | render 200-message payload p50 | 0.171 (0.170-0.174) | 0.221 (0.220-0.228) | 0.432 (0.429-0.496) |
| render | render 200-message payload p99 | 0.183 (0.177-0.203) | 0.480 (0.469-0.589) | 1.080 (1.060-1.401) |

## Throughput: N concurrent chats, 5 asks each, 50 ms mock delay

| Chats | ideal req/s | rust | ruby+yjit (Async fibers) | ruby+yjit threads | rust RSS MiB | ruby fibers RSS MiB | ruby threads RSS MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 19 (19-19) | 17 (17-17) | 17 (17-18) | 24 (24-24) | 61 (61-61) | 60 (58-60) |
| 10 | 200 | 191 (189-192) | 147 (145-148) | 132 (129-137) | 26 (26-26) | 66 (66-66) | 66 (64-66) |
| 100 | 2,000 | 1,775 (1,755-1,792) | 1,090 (981-1,146) | 939 (937-957) | 37 (37-37) | 71 (71-75) | 72 (72-73) |
| 1000 | 20,000 | 13,295 (13,200-13,716) | 2,680 (2,659-2,749) | 2,008 (2,000-2,044) | 148 (148-149) | 91 (91-93) | 165 (164-165) |

## Memory: RSS (MiB) with N chats in flight

| Chats | rust baseline | rust in flight | ruby fibers baseline | ruby fibers in flight | ruby threads in flight |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.3 (23.3-23.4) | 23.7 (23.5-23.7) | 59.3 (57.3-59.3) | 59.5 (57.6-59.6) | 59.5 (57.5-59.5) |
| 100 | 23.3 (23.2-23.4) | 30.0 (30.0-30.1) | 59.3 (57.3-59.4) | 66.7 (65.2-66.8) | 69.4 (67.6-69.5) |

## One long chat: sequential asks in one conversation, 0 ms delay

| Impl | asks | elapsed s | RSS MiB |
|---|---:|---:|---:|
| ruby+yjit | 300 | 0.44 (0.44-0.60) | 72 (69-72) |
| rust | 300 | 0.26 (0.24-0.28) | 35 (35-35) |
| rust strip_raw | 300 | 0.26 (0.24-0.29) | 28 (28-28) |
