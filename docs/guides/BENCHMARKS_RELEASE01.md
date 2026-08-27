# SimpleNAV Release 01 Benchmarks

[Back to the main README](../../README.md) · [中文](BENCHMARKS_RELEASE01_ZH.md)

## Overview

| Benchmark | Split / Task | SimpleNAV result |
| --- | --- | --- |
| OpenFly | Seen | NE 37.1 m · SR 52.8% · OSR 74.2% · SPL 51.0% |
| TravelUAV | Test Seen · Full | NE 85.6 m · SR 22.4 · OSR 55.1 · SPL 20.5 |
| AerialVLN-S | Val Seen | NE 126.0 m · SR 8.4 · OSR 18.9 · SDTW 3.4 |
| R2R-CE | Val-Unseen | NE 4.7 m · OS 55.9 · SR 49.2 · SPL 45.8 |
| RxR-CE | Val-Unseen | NE 4.6 m · SR 58.4 · SPL 52.2 · nDTW 74.6 |
| EVT-Bench | STT | SR 82.8 · TR 93.5 · CR 1.2 |
| EVT-Bench | DT | SR 45.2 · TR 76.7 · CR 6.1 |
| EVT-Bench | AT | SR 39.7 · TR 79.2 · CR 3.8 |

## OpenFly Seen

| Method | Source | NE↓ (m) | SR↑ | OSR↑ | SPL↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Random | Baseline | 242.0 | 0.7% | 0.8% | 0.0% |
| Seq2Seq | [Baseline code][vln-ce] | 205.0 | 2.9% | 24.3% | 2.6% |
| CMA | [Baseline code][vln-ce] | 161.0 | 5.4% | 28.1% | 4.8% |
| See-Point-Fly | — | — | — | — | — |
| AerialVLN | [Benchmark code][aerialvln-benchmark] | 139.0 | 7.5% | 3.0% | 6.8% |
| NaVid | [Code + weights][navid] | 153.0 | 13.0% | 38.2% | 11.6% |
| NaVila | — | 132.0 | 20.3% | 53.5% | 17.8% |
| OpenFly-Agent | — | 93.0 | 34.3% | 64.3% | 24.9% |
| **SimpleNAV (single-dataset)** | **Repository** | **37.1** | **52.8%** | **74.2%** | **51.0%** |

## TravelUAV Test Seen

### Full

| Method | Source | NE↓ (m) | SR↑ | OSR↑ | SPL↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Human | Baseline | 14.2 | 94.5 | 94.5 | 77.9 |
| Random Action | Baseline | 222.2 | 0.1 | 0.2 | 0.1 |
| Fixed Action | Baseline | 188.6 | 2.3 | 8.2 | 1.4 |
| CMA | [Baseline code][vln-ce] | 135.7 | 8.4 | 18.7 | 7.9 |
| TravelUAV | [Code + weights][traveluav] | 106.3 | 16.1 | 44.3 | 14.3 |
| TravelUAV-DA | [Code + weights][traveluav] | 98.7 | 17.5 | 48.9 | 15.8 |
| NavFoM | [Project page only][navfom] | 93.1 | 29.2 | 49.2 | 25.0 |
| **LongFly** | [Paper only][longfly] | **60.0** | 36.4 | **65.9** | 31.1 |
| **AerialVLA** | [Code + weights][aerialvla] | 65.9 | **48.0** | 57.7 | **38.6** |
| **SimpleNAV (single-dataset)** | **Repository** | 85.6 | 22.4 | 55.1 | 20.5 |

### Easy

| Method | Source | NE↓ (m) | SR↑ | OSR↑ | SPL↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Human | Baseline | 11.7 | 95.4 | 95.4 | 76.2 |
| Random Action | Baseline | 142.1 | 0.3 | 0.4 | 0.1 |
| Fixed Action | Baseline | 121.4 | 3.5 | 11.5 | 2.1 |
| CMA | [Baseline code][vln-ce] | 84.9 | 11.5 | 24.5 | 10.7 |
| TravelUAV | [Code + weights][traveluav] | 68.8 | 18.8 | 47.6 | 16.4 |
| TravelUAV-DA | [Code + weights][traveluav] | 66.4 | 20.3 | 51.2 | 18.1 |
| NavFoM | [Project page only][navfom] | 59.0 | 32.9 | 53.2 | 27.9 |
| **LongFly** | [Paper only][longfly] | **38.1** | 38.5 | **71.9** | 31.2 |
| **AerialVLA** | [Code + weights][aerialvla] | 43.8 | **49.3** | 61.3 | **37.1** |
| **SimpleNAV (single-dataset)** | **Repository** | 60.0 | 22.8 | 56.9 | 21.0 |

### Hard

| Method | Source | NE↓ (m) | SR↑ | OSR↑ | SPL↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Human | Baseline | 17.2 | 93.4 | 93.4 | 79.8 |
| Random Action | Baseline | 320.1 | 0.0 | 0.0 | 0.0 |
| Fixed Action | Baseline | 270.7 | 0.8 | 4.1 | 0.5 |
| CMA | [Baseline code][vln-ce] | 197.8 | 4.6 | 11.7 | 4.5 |
| TravelUAV | [Code + weights][traveluav] | 152.0 | 12.8 | 40.2 | 11.8 |
| TravelUAV-DA | [Code + weights][traveluav] | 138.0 | 14.0 | 46.0 | 12.9 |
| NavFoM | [Project page only][navfom] | 143.8 | 23.6 | 43.4 | 20.8 |
| **LongFly** | [Paper only][longfly] | **85.2** | 33.9 | **58.9** | 30.9 |
| **AerialVLA** | [Code + weights][aerialvla] | 93.2 | **46.3** | 53.2 | **40.3** |
| **SimpleNAV (single-dataset)** | **Repository** | 118.3 | 21.9 | 52.8 | 19.9 |

## AerialVLN-S Val Seen

| Method | Source | NE↓ | SR↑ | OSR↑ | SDTW↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| Random | Baseline | 109.6 | 0.0 | 0.0 | 0.0 |
| Action Sampling | [Benchmark code][aerialvln-benchmark] | 213.8 | 0.9 | 5.7 | 0.3 |
| LingUNet | — | 383.8 | 0.6 | 6.9 | 0.2 |
| Seq2Seq | [Benchmark code][aerialvln-benchmark] | 146.0 | 4.8 | 19.8 | 1.6 |
| CMA | [Benchmark code][aerialvln-benchmark] | 121.0 | 3.0 | 23.2 | 0.6 |
| **Seq2Seq-DA** | [Benchmark code][aerialvln-benchmark] | **85.5** | **9.9** | 24.1 | **4.5** |
| **CMA-DA** | [Benchmark code][aerialvln-benchmark] | 92.2 | **9.9** | **26.5** | 3.7 |
| LAG | — | 90.2 | 7.2 | 15.7 | 2.4 |
| **SimpleNAV (single-dataset)** | **Repository** | 126.0 | 8.4 | 18.9 | 3.4 |

## R2R-CE Val-Unseen

| Method | Source | Training data | Model size | Input | NE↓ (m) | OS↑ | SR↑ | SPL↑ |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen-RobotNav-4B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | 4B | Single-view | 4.2 | 73.6 | 66.9 | 60.5 |
| Qwen-RobotNav-8B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | 8B | Single-view | 4.4 | 72.7 | 65.7 | 59.6 |
| Qwen-RobotNav-4B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | 4B | Panoramic | 3.8 | 77.2 | 69.5 | 63.6 |
| **Qwen-RobotNav-8B** | [Report repo; no code/weights][qwen-robotnav] | 15.6M | 8B | Panoramic | **3.5** | **78.5** | **72.1** | **66.6** |
| ABot-N0 | [Report repo; no code/weights][abot-n0] | 21.9M | 4B | Panoramic RGB | 3.8 | 70.8 | 66.4 | 63.9 |
| InternVLA-N1 (S2) | [Code + weights][internvla-n1] | >5M | 8B | Single-view RGB | 4.9 | 60.6 | 55.4 | 52.1 |
| InternVLA-N1 (S1+S2) | [Code + weights][internvla-n1] | — | 8B | Single-view RGB + Depth | 4.8 | 63.3 | 58.2 | 54.0 |
| NavFoM | [Project page only][navfom] | 12.7M | 7B | Single-view RGB | 5.0 | 64.9 | 56.2 | 51.2 |
| NavFoM | [Project page only][navfom] | 12.7M | 7B | Four-view RGB | 4.6 | 72.1 | 61.7 | 55.3 |
| Uni-NaVid | [Code + weights][uni-navid] | 5.9M | 7B | Single-view RGB | 5.6 | 53.3 | 47.0 | 42.7 |
| **SimpleNAV (single-dataset)** | **Repository** | 1.9M | 5.3B | Four-view RGB | 4.7 | 55.9 | 49.2 | 45.8 |

## RxR-CE Val-Unseen

| Method | Source | Training data | Model size | Input | NE↓ (m) | SR↑ | SPL↑ | nDTW↑ |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen-RobotNav-4B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | 4B | Single-view | 4.2 | 71.3 | 61.5 | 68.6 |
| Qwen-RobotNav-8B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | 8B | Single-view | 4.2 | 73.4 | 63.5 | 69.9 |
| Qwen-RobotNav-4B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | 4B | Panoramic | 3.8 | 75.2 | 65.0 | 71.9 |
| **Qwen-RobotNav-8B** | [Report repo; no code/weights][qwen-robotnav] | 15.6M | 8B | Panoramic | **3.6** | **76.5** | **65.7** | 72.5 |
| ABot-N0 | [Report repo; no code/weights][abot-n0] | 16.9M expert + 5.0M reasoning | 4B | Panoramic RGB | 3.8 | 69.3 | 60.0 | — |
| InternVLA-N1 (S2) | [Code + weights][internvla-n1] | >5M | 8B | Single-view RGB | 6.4 | 49.5 | 41.8 | 62.6 |
| InternVLA-N1 (S1+S2) | [Code + weights][internvla-n1] | — | 8B | Single-view RGB + Depth | 5.9 | 53.5 | 46.1 | 65.3 |
| NavFoM | [Project page only][navfom] | 12.7M | 7B | Single-view RGB | 5.5 | 57.4 | 49.4 | 60.2 |
| NavFoM | [Project page only][navfom] | 12.7M | 7B | Four-view RGB | 4.7 | 64.4 | 56.2 | 65.8 |
| Uni-NaVid | [Code + weights][uni-navid] | 5.9M | 7B | Single-view RGB | 6.2 | 48.7 | 40.9 | — |
| **SimpleNAV (single-dataset)** | **Repository** | 1.9M | 5.3B | Four-view RGB | 4.6 | 58.4 | 52.2 | **74.6** |

## EVT-Bench

### Model settings

| Method | Source | Training data | Model size | Input |
| --- | --- | --- | --- | --- |
| Qwen-RobotNav-4B / 8B | [Report repo; no code/weights][qwen-robotnav] | 15.6M | 4B / 8B | Single-view |
| ABot-N0 | [Report repo; no code/weights][abot-n0] | 21.9M | 4B | Single-view |
| VLingNav / SFT | [Project page only][vlingnav] | 4.5M | 7B | Single-view |
| TrackVLA | [Benchmark code only][trackvla] | 1.7M | 7B | Single-view |
| TrackVLA++ | [Project page only][trackvla-pp] | 2M | 7B | Single-view / Four-view |
| NavFoM | [Project page only][navfom] | 12.7M | 7B | Single-view / Four-view |
| Uni-NaVid | [Code + weights][uni-navid] | — | 7B | Single-view |
| **SimpleNAV (single-dataset)** | **Repository** | 2.4M | 5.3B | Single-view |

### STT

| Method | Training data | Model size | Input | SR↑ | TR↑ | CR↓ |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Qwen-RobotNav-4B | 15.6M | 4B | Single-view | 77.4 | 90.0 | 6.4 |
| Qwen-RobotNav-8B | 15.6M | 8B | Single-view | 78.6 | 89.7 | 5.7 |
| ABot-N0 | 21.9M | 4B | Single-view | 86.9 | 87.6 | 8.5 |
| VLingNav | 4.5M | 7B | Single-view | 88.4 | 81.2 | 2.1 |
| **VLingNav (SFT)** | 4.5M | 7B | Single-view | 87.2 | 78.9 | **1.2** |
| TrackVLA | 1.7M | 7B | Single-view | 85.1 | 78.6 | 1.7 |
| TrackVLA++ | 2M | 7B | Single-view | 86.0 | 81.0 | 2.1 |
| **TrackVLA++** | 2M | 7B | Four-view | **90.9** | 82.7 | 1.5 |
| NavFoM | 12.7M | 7B | Single-view | 85.0 | 80.5 | — |
| NavFoM | 12.7M | 7B | Four-view | 88.4 | 80.7 | — |
| Uni-NaVid | — | 7B | Single-view | 25.7 | 39.5 | 41.9 |
| **SimpleNAV (single-dataset)** | 2.4M | 5.3B | Single-view | 82.8 | **93.5** | **1.2** |

### DT

| Method | Training data | Model size | Input | SR↑ | TR↑ | CR↓ |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Qwen-RobotNav-4B | 15.6M | 4B | Single-view | — | — | — |
| Qwen-RobotNav-8B | 15.6M | 8B | Single-view | — | — | — |
| ABot-N0 | 21.9M | 4B | Single-view | 66.7 | 75.4 | 11.6 |
| VLingNav | 4.5M | 7B | Single-view | 67.6 | 73.5 | 5.5 |
| VLingNav (SFT) | 4.5M | 7B | Single-view | 66.1 | 69.7 | 4.8 |
| TrackVLA | 1.7M | 7B | Single-view | 57.6 | 63.2 | 5.8 |
| TrackVLA++ | 2M | 7B | Single-view | 66.5 | 68.8 | 4.7 |
| **TrackVLA++** | 2M | 7B | Four-view | **74.0** | 73.7 | **3.5** |
| NavFoM | 12.7M | 7B | Single-view | 61.4 | 68.2 | — |
| NavFoM | 12.7M | 7B | Four-view | 62.0 | 67.9 | — |
| Uni-NaVid | — | 7B | Single-view | 11.3 | 27.4 | 43.5 |
| **SimpleNAV (single-dataset)** | 2.4M | 5.3B | Single-view | 45.2 | **76.7** | 6.0 |

### AT

| Method | Training data | Model size | Input | SR↑ | TR↑ | CR↓ |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Qwen-RobotNav-4B | 15.6M | 4B | Single-view | — | — | — |
| Qwen-RobotNav-8B | 15.6M | 8B | Single-view | — | — | — |
| **ABot-N0** | 21.9M | 4B | Single-view | **67.3** | **79.5** | 7.0 |
| VLingNav | 4.5M | 7B | Single-view | — | — | — |
| VLingNav (SFT) | 4.5M | 7B | Single-view | — | — | — |
| TrackVLA | 1.7M | 7B | Single-view | 50.2 | 63.7 | 17.1 |
| TrackVLA++ | 2M | 7B | Single-view | 51.2 | 63.4 | 15.9 |
| TrackVLA++ | 2M | 7B | Four-view | 55.9 | 63.8 | 15.1 |
| NavFoM | 12.7M | 7B | Single-view | — | — | — |
| NavFoM | 12.7M | 7B | Four-view | — | — | — |
| Uni-NaVid | — | 7B | Single-view | 8.3 | 28.6 | 43.7 |
| **SimpleNAV (single-dataset)** | 2.4M | 5.3B | Single-view | 39.7 | 79.2 | **3.8** |

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
