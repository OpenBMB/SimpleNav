# SimpleNAV Release 01 Benchmarks

[Back to the main README](../../README.md) · [中文](BENCHMARKS_RELEASE01_ZH.md)

`—` means not reported. Artifact labels reflect official project pages checked for the `xw` comparison snapshot on 2026-08-16. **Full stack** denotes the data, training, evaluation, and configuration source published in this repository.

## Overview

| Benchmark | Split / Task | SimpleNAV result |
| --- | --- | --- |
| OpenFly | Seen | NE 37.12 m · SR 52.85% · OSR 74.15% · SPL 50.96% |
| TravelUAV | Test Seen · Full | NE 85.61 m · SR 22.42 · OSR 55.08 · SPL 20.51 |
| AerialVLN-S | Val Seen | NE 126 m · SR 8.4 · OSR 18.92 · SDTW 3.4 |
| R2R-CE | Val-Unseen | NE 4.65 m · OS 55.93 · SR 49.18 · SPL 45.82 |
| RxR-CE | Val-Unseen | NE 4.62 m · SR 58.44 · SPL 52.17 · nDTW 74.60 |
| EVT-Bench | STT | SR 89.31 · TR 96.08 · CR 1.09 |
| EVT-Bench | DT | SR 45.20 · TR 76.67 · CR 6.05 |
| EVT-Bench | AT | SR 39.72 · TR 79.20 · CR 3.77 |

## OpenFly Seen

| Method | Open artifacts | NE↓ (m) | SR↑ | OSR↑ | SPL↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Random | Baseline | 242 | 0.7% | 0.8% | 0% |
| Seq2Seq | [Baseline code][vln-ce] | 205 | 2.9% | 24.3% | 2.6% |
| CMA | [Baseline code][vln-ce] | 161 | 5.4% | 28.1% | 4.8% |
| See-Point-Fly | Release not recorded | — | — | — | — |
| AerialVLN | [Benchmark code][aerialvln-benchmark] | 139 | 7.5% | 30.0% | 6.8% |
| NaVid | [Code + weights][navid] | 153 | 13.0% | 38.2% | 11.6% |
| NaVila | Release not recorded | 132 | 20.3% | 53.5% | 17.8% |
| OpenFly-Agent | Release not recorded | 93 | 34.3% | 64.3% | 24.9% |
| **SimpleNAV (single-dataset)** | **Full stack** | **37.12** | **52.85%** | **74.15%** | **50.96%** |

## TravelUAV Test Seen

### Full

| Method | Open artifacts | NE↓ (m) | SR↑ | OSR↑ | SPL↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Human | Baseline | 14.15 | 94.51 | 94.51 | 77.84 |
| Random Action | Baseline | 222.20 | 0.14 | 0.21 | 0.07 |
| Fixed Action | Baseline | 188.61 | 2.27 | 8.16 | 1.40 |
| CMA | [Baseline code][vln-ce] | 135.73 | 8.37 | 18.72 | 7.90 |
| TravelUAV | [Code + weights][traveluav] | 106.28 | 16.10 | 44.26 | 14.30 |
| TravelUAV-DA | [Code + weights][traveluav] | 98.66 | 17.45 | 48.87 | 15.76 |
| NavFoM | [Project page only][navfom] | 93.05 | 29.17 | 49.24 | 25.03 |
| LongFly | [Paper only][longfly] | 60.02 | 36.39 | 65.87 | 31.07 |
| AerialVLA | [Code + weights][aerialvla] | 65.88 | 47.96 | 57.69 | 38.54 |
| **SimpleNAV (single-dataset)** | **Full stack** | **85.61** | **22.42** | **55.08** | **20.51** |

### Easy

| Method | Open artifacts | NE↓ (m) | SR↑ | OSR↑ | SPL↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Human | Baseline | 11.68 | 95.44 | 95.44 | 76.19 |
| Random Action | Baseline | 142.07 | 0.26 | 0.39 | 0.13 |
| Fixed Action | Baseline | 121.36 | 3.48 | 11.48 | 2.14 |
| CMA | [Baseline code][vln-ce] | 84.89 | 11.48 | 24.52 | 10.68 |
| TravelUAV | [Code + weights][traveluav] | 68.78 | 18.84 | 47.61 | 16.39 |
| TravelUAV-DA | [Code + weights][traveluav] | 66.40 | 20.26 | 51.23 | 18.10 |
| NavFoM | [Project page only][navfom] | 58.98 | 32.91 | 53.16 | 27.87 |
| LongFly | [Paper only][longfly] | 38.10 | 38.52 | 71.90 | 31.24 |
| AerialVLA | [Code + weights][aerialvla] | 43.76 | 49.30 | 61.30 | 37.14 |
| **SimpleNAV (single-dataset)** | **Full stack** | **59.96** | **22.80** | **56.87** | **21.01** |

### Hard

| Method | Open artifacts | NE↓ (m) | SR↑ | OSR↑ | SPL↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Human | Baseline | 17.16 | 93.37 | 93.37 | 79.85 |
| Random Action | Baseline | 320.12 | 0.00 | 0.00 | 0.00 |
| Fixed Action | Baseline | 270.69 | 0.79 | 4.09 | 0.49 |
| CMA | [Baseline code][vln-ce] | 197.77 | 4.57 | 11.65 | 4.51 |
| TravelUAV | [Code + weights][traveluav] | 152.04 | 12.76 | 40.16 | 11.76 |
| TravelUAV-DA | [Code + weights][traveluav] | 138.04 | 14.02 | 45.98 | 12.90 |
| NavFoM | [Project page only][navfom] | 143.83 | 23.58 | 43.40 | 20.80 |
| LongFly | [Paper only][longfly] | 85.20 | 33.94 | 58.94 | 30.88 |
| AerialVLA | [Code + weights][aerialvla] | 93.16 | 46.30 | 53.23 | 40.26 |
| **SimpleNAV (single-dataset)** | **Full stack** | **118.28** | **21.95** | **52.81** | **19.88** |

## AerialVLN-S Val Seen

| Method | Open artifacts | NE↓ | SR↑ | OSR↑ | SDTW↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Random | Baseline | 109.6 | 0 | 0 | 0 |
| Action Sampling | [Benchmark code][aerialvln-benchmark] | 213.8 | 0.9 | 5.7 | 0.3 |
| LingUNet | Release not recorded | 383.8 | 0.6 | 6.9 | 0.2 |
| Seq2Seq | [Benchmark code][aerialvln-benchmark] | 146 | 4.8 | 19.8 | 1.6 |
| CMA | [Benchmark code][aerialvln-benchmark] | 121 | 3 | 23.2 | 0.6 |
| Seq2Seq-DA | [Benchmark code][aerialvln-benchmark] | 85.5 | 9.9 | 24.1 | 4.5 |
| CMA-DA | [Benchmark code][aerialvln-benchmark] | 92.2 | 9.9 | 26.5 | 3.7 |
| LAG | Release not recorded | 90.2 | 7.2 | 15.7 | 2.4 |
| **SimpleNAV (single-dataset)** | **Full stack** | **126** | **8.4** | **18.92** | **3.4** |

## R2R-CE Val-Unseen

| Method | Open artifacts | Training data | Input | NE↓ (m) | OS↑ | SR↑ | SPL↑ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen-RobotNav-4B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | Single-view | 4.22 | 73.6 | 66.9 | 60.5 |
| Qwen-RobotNav-8B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | Single-view | 4.36 | 72.7 | 65.7 | 59.6 |
| Qwen-RobotNav-4B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | Panoramic | 3.80 | 77.2 | 69.5 | 63.6 |
| Qwen-RobotNav-8B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | Panoramic | 3.53 | 78.5 | 72.1 | 66.6 |
| ABot-N0 | [Report repo; no code/weights][abot-n0] | 21.9M | Panoramic RGB | 3.78 | 70.8 | 66.4 | 63.9 |
| InternVLA-N1 (S2) | [Code + weights][internvla-n1] | >5M | Single-view RGB | 4.89 | 60.6 | 55.4 | 52.1 |
| InternVLA-N1 (S1+S2) | [Code + weights][internvla-n1] | Not reported | Single-view RGB + Depth | 4.83 | 63.3 | 58.2 | 54.0 |
| NavFoM | [Project page only][navfom] | 12.7M | Single-view RGB | 5.01 | 64.9 | 56.2 | 51.2 |
| NavFoM | [Project page only][navfom] | 12.7M | Four-view RGB | 4.61 | 72.1 | 61.7 | 55.3 |
| Uni-NaVid | [Code + weights][uni-navid] | 5.9M | Single-view RGB | 5.58 | 53.3 | 47.0 | 42.7 |
| **SimpleNAV (single-dataset)** | **Full stack** | **1.9M** | **Four-view RGB** | **4.65** | **55.93** | **49.18** | **45.82** |

## RxR-CE Val-Unseen

| Method | Open artifacts | Training data | Input | NE↓ (m) | SR↑ | SPL↑ | nDTW↑ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen-RobotNav-4B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | Single-view | 4.15 | 71.3 | 61.5 | 68.6 |
| Qwen-RobotNav-8B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | Single-view | 4.16 | 73.4 | 63.5 | 69.9 |
| Qwen-RobotNav-4B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | Panoramic | 3.80 | 75.2 | 65.0 | 71.9 |
| Qwen-RobotNav-8B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | Panoramic | 3.58 | 76.5 | 65.7 | 72.5 |
| ABot-N0 | [Report repo; no code/weights][abot-n0] | 16.9M expert + 5.0M reasoning | Panoramic RGB | 3.83 | 69.3 | 60.0 | — |
| InternVLA-N1 (S2) | [Code + weights][internvla-n1] | >5M | Single-view RGB | 6.41 | 49.5 | 41.8 | 62.6 |
| InternVLA-N1 (S1+S2) | [Code + weights][internvla-n1] | Not reported | Single-view RGB + Depth | 5.91 | 53.5 | 46.1 | 65.3 |
| NavFoM | [Project page only][navfom] | 12.7M | Single-view RGB | 5.51 | 57.4 | 49.4 | 60.2 |
| NavFoM | [Project page only][navfom] | 12.7M | Four-view RGB | 4.74 | 64.4 | 56.2 | 65.8 |
| Uni-NaVid | [Code + weights][uni-navid] | 5.9M | Single-view RGB | 6.24 | 48.7 | 40.9 | — |
| **SimpleNAV (single-dataset)** | **Full stack** | **1.9M** | **Four-view RGB** | **4.62** | **58.44** | **52.17** | **74.60** |

## EVT-Bench

### Model settings

| Method | Open artifacts | Training data | Input |
| --- | --- | --- | --- |
| Qwen-RobotNav-4B / 8B | [Report repo; no code/weights][qwen-robotnav] | 1.5M | Single-view |
| ABot-N0 | [Report repo; no code/weights][abot-n0] | 4.0M | Single-view |
| VLingNav / SFT | [Project page only][vlingnav] | 855K | Single-view |
| TrackVLA | [Benchmark code only][trackvla] | 855K | Single-view |
| TrackVLA++ | [Project page only][trackvla-pp] | 1M | Single-view / Four-view |
| NavFoM | [Project page only][navfom] | 897K | Single-view / Four-view |
| Uni-NaVid | [Code + weights][uni-navid] | Not disclosed | Single-view |
| **SimpleNAV (single-dataset)** | **Full stack** | **2.4M** | **Single-view** |

### STT

| Method | Input | SR↑ | TR↑ | CR↓ |
| --- | --- | ---: | ---: | ---: |
| Qwen-RobotNav-4B | Single-view | 77.4 | 90.0 | 6.40 |
| Qwen-RobotNav-8B | Single-view | 78.6 | 89.7 | 5.70 |
| ABot-N0 | Single-view | 86.9 | 87.6 | 8.54 |
| VLingNav | Single-view | 88.4 | 81.2 | 2.07 |
| VLingNav (SFT) | Single-view | 87.2 | 78.9 | 1.23 |
| TrackVLA | Single-view | 85.1 | 78.6 | 1.65 |
| TrackVLA++ | Single-view | 86.0 | 81.0 | 2.10 |
| TrackVLA++ | Four-view | 90.9 | 82.7 | 1.50 |
| NavFoM | Single-view | 85.0 | 80.5 | — |
| NavFoM | Four-view | 88.4 | 80.7 | — |
| Uni-NaVid | Single-view | 25.7 | 39.5 | 41.9 |
| **SimpleNAV (single-dataset)** | **Single-view** | **89.31** | **96.08** | **1.09** |

### DT

| Method | Input | SR↑ | TR↑ | CR↓ |
| --- | --- | ---: | ---: | ---: |
| Qwen-RobotNav-4B | Single-view | — | — | — |
| Qwen-RobotNav-8B | Single-view | — | — | — |
| ABot-N0 | Single-view | 66.7 | 75.4 | 11.6 |
| VLingNav | Single-view | 67.6 | 73.5 | 5.51 |
| VLingNav (SFT) | Single-view | 66.1 | 69.7 | 4.78 |
| TrackVLA | Single-view | 57.6 | 63.2 | 5.80 |
| TrackVLA++ | Single-view | 66.5 | 68.8 | 4.71 |
| TrackVLA++ | Four-view | 74.0 | 73.7 | 3.51 |
| NavFoM | Single-view | 61.4 | 68.2 | — |
| NavFoM | Four-view | 62.0 | 67.9 | — |
| Uni-NaVid | Single-view | 11.3 | 27.4 | 43.5 |
| **SimpleNAV (single-dataset)** | **Single-view** | **45.20** | **76.67** | **6.05** |

### AT

| Method | Input | SR↑ | TR↑ | CR↓ |
| --- | --- | ---: | ---: | ---: |
| Qwen-RobotNav-4B | Single-view | — | — | — |
| Qwen-RobotNav-8B | Single-view | — | — | — |
| ABot-N0 | Single-view | 67.3 | 79.5 | 7.05 |
| VLingNav | Single-view | — | — | — |
| VLingNav (SFT) | Single-view | — | — | — |
| TrackVLA | Single-view | 50.2 | 63.7 | 17.1 |
| TrackVLA++ | Single-view | 51.2 | 63.4 | 15.9 |
| TrackVLA++ | Four-view | 55.9 | 63.8 | 15.1 |
| NavFoM | Single-view | — | — | — |
| NavFoM | Four-view | — | — | — |
| Uni-NaVid | Single-view | 8.26 | 28.6 | 43.7 |
| **SimpleNAV (single-dataset)** | **Single-view** | **39.72** | **79.20** | **3.77** |

## Metrics

- **NE:** Navigation Error; lower is better.
- **SR / OSR / OS:** Success Rate / Oracle Success Rate; higher is better.
- **SPL:** Success weighted by Path Length; higher is better.
- **nDTW / SDTW:** trajectory-matching metrics; higher is better.
- **TR / CR:** EVT-Bench task metrics.

[qwen-robotnav]: https://github.com/QwenLM/Qwen-RobotNav
[abot-n0]: https://github.com/amap-cvlab/ABot-Navigation/tree/ABot-N0
[internvla-n1]: https://github.com/InternRobotics/InternNav
[navfom]: https://pku-epic.github.io/NavFoM-Web/
[uni-navid]: https://github.com/jzhzhang/Uni-NaVid
[vln-ce]: https://github.com/jacobkrantz/VLN-CE
[navid]: https://github.com/jzhzhang/NaVid-VLN-CE
[traveluav]: https://github.com/prince687028/TravelUAV
[longfly]: https://arxiv.org/abs/2512.22010
[aerialvla]: https://github.com/XuPeng23/AeroVLA
[aerialvln-benchmark]: https://github.com/AirVLN/AirVLN
[vlingnav]: https://wsakobe.github.io/VLingNav-web/
[trackvla]: https://github.com/wsakobe/TrackVLA
[trackvla-pp]: https://pku-epic.github.io/TrackVLA-plus-plus-Web/
