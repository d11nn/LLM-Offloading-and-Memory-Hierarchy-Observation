---
title: 'Assignment 3 - LLM Offloading and Memory Hierarchy Observation'

---

# Assignment 3 - LLM Offloading and Memory Hierarchy Observation
**[NYCU [CSIC30155] Memory and Storage System](https://timetable.nycu.edu.tw/?r=main/crsoutline&Acy=114&Sem=2&CrsNo=535512&lang=zh-tw) by Prof. [Chun-Feng Wu](https://www.cs.nycu.edu.tw/members/detail/cfwu417)**

<font color="#f00"> **Due Date: 23:59, Monday, June 1, 2026** </font>

---
**Table of Contents**

[TOC]

---

## Objectives
- Use **FlexGen** to observe workload behavior across the GPU / CPU RAM / Disk three-tier memory hierarchy
- Compare how different offloading strategies affect throughput, memory usage, and I/O patterns

## Introduction

[FlexGen](https://github.com/FMInference/FlexLLMGen) is an inference framework that distributes a large language model's weights, KV cache, and activations across the three tiers GPU / CPU RAM / Disk. The CLI flag `--percent` controls how these three tensor types are distributed across the tiers; **for simplicity, this assignment only adjusts the weight distribution and keeps KV cache and activation entirely on GPU**.

:::warning
**No AI background is required for this assignment**. FlexGen's behavior can be reduced to: every output token requires the entire model (~2.4 GB FP16 weights) to "pass through GPU once" — the only difference between configurations is where these weights normally reside, and where each step has to fetch them from. We focus only on system-level behavior: which tier holds what, how much is moved, what the I/O pattern looks like, and where the bottleneck lies.
:::

We use [`facebook/opt-1.3b`](https://huggingface.co/facebook/opt-1.3b) throughout.

## Environment Setup

Requires **NVIDIA GPU + CUDA**. Students without a GPU should use the "Colab Path" (`lab03_colab.ipynb` is in the assignment zip).

### Recommended Hardware

GPU VRAM ≥ 4 GB, CPU RAM ≥ 16 GB, free disk ≥ 8 GB.

### Installation

Create an empty folder as your **assignment root directory** (run all subsequent commands from here), then download and extract the [assignment zip](https://drive.google.com/file/d/1Y5fZ32iv0DDKO-2JkSxe7rOLxfAQJNbw/view?usp=sharing) into it. Any Python + PyTorch (CUDA build) environment that runs FlexGen works; conda is shown below. Fall back to the provided `environment.yml` if you hit environment issues.

```shell
cd <your_assignment_root>

# 1. Create environment (conda example)
conda env create -f environment.yml
conda activate flexgen-lab

# 2. Clone and install FlexGen, then return to assignment root
git clone https://github.com/FMInference/FlexLLMGen.git
cd FlexLLMGen && pip install -e . && cd ..

# 3. Create logs and offload directories under the assignment root
mkdir -p logs flexgen_offload

# 4. Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Model Download

On first run, FlexGen automatically downloads OPT-1.3B and converts it to numpy format under `~/opt_weights/opt-1.3b-np/` (~2.8 GB).

### Colab Path (for students without a GPU)

1. Upload `lab03_colab.ipynb`to [Google Colab](https://colab.research.google.com/)
2. **Runtime → Change runtime type → GPU (T4)** (free tier is fine)
3. Run cells top-to-bottom. The notebook includes boilerplate and a Q1 dp(1) example; replicate the pattern for the rest.

:::info
Colab T4's disk is shared cloud storage (an order of magnitude slower than a local NVMe); local vs Colab throughput is not directly comparable, so state which path you used in your report.
:::

### Environment Verification

After running Q1 datapoint (1) (100% GPU baseline), compare against `q1_expected_output.txt`. Normal output with no error messages is sufficient.

Common errors:

| Error | Resolution |
|---|---|
| `RuntimeError: torch.cuda.Stream requires CUDA support` | PyTorch lacks CUDA support; reinstall the cu124 wheel |
| `Killed` | OOM; switch to Colab |
| First run is very slow | Normal — FlexGen is downloading + converting (~3-5 min) |
| `TypedStorage is deprecated` / `reduce_op is deprecated` warnings | Ignore — PyTorch 2.6 deprecation warnings, do not affect execution |

---

## Exercises

:::info
- All commands fix `--prompt-len 128 --gen-len 16`; **token = one output unit**.
- `--percent W_g W_c K_g K_c A_g A_c`: GPU and CPU percentages for weight / KV cache / activation; whatever doesn't sum to 100% within a category auto-offloads to disk. **This assignment only adjusts the first two** (weight split); the rest is fixed at `100 0 100 0`.
:::

### Q1. (30%) Progressive Offload Sweep

With `--gpu-batch-size 4 --prompt-len 128 --gen-len 16` fixed, run 5 weight distributions:

| Setting | `--percent` | Weight Distribution |
|---|---|---|
| (1) | `100 0 100 0 100 0` | 100% GPU |
| (2) | `50 50 100 0 100 0` | 50% GPU + 50% CPU |
| (3) | `50 0 100 0 100 0` | 50% GPU + 50% Disk |
| (4) | `0 50 100 0 100 0` | 50% CPU + 50% Disk |
| (5) | `0 0 100 0 100 0` | 100% Disk |

```shell
python -m flexllmgen.flex_opt --model facebook/opt-1.3b \
    --percent <one of the 5 above> --gpu-batch-size 4 \
    --prompt-len 128 --gen-len 16 --offload-dir flexgen_offload \
    2>&1 | tee logs/q1_<1-5>.log
```

<font color="blue">**Questions:**</font>
1. (10%) Tabulate **total throughput** and **peak gpu mem** for the 5 datapoints.
2. (20%) Complete the table below:

   | Boundary | Ratio | Dominant Bottleneck (PCIe / Disk) |
   |---|---|---|
   | (1)→(2) | `(1) / (2)` = __ | __ |
   | (2)→(3) | `(2) / (3)` = __ | __ |
   | (3)→(4) | `(3) / (4)` = __ | __ |
   | (4)→(5) | `(4) / (5)` = __ | __ |

   > "Disk" refers to your storage medium (NVMe / SATA SSD / Colab overlay / etc.). Determine its bandwidth order of magnitude based on your actual hardware.

   After filling in the table, answer:

   > **Front pair** = (1)→(2), (2)→(3); **back pair** = (3)→(4), (4)→(5).

   - (a) **Within-class comparison**: Are the two PCIe boundary ratios close to each other? What about the two Disk boundary ratios? If there's a gap, is the front-pair ratio larger or the back-pair ratio larger? Why?
   - (b) **Cross-class within-pair comparison**: In the front pair (1)→(2) PCIe vs (2)→(3) Disk, and in the back pair (3)→(4) PCIe vs (4)→(5) Disk, **within each pair**, which has the larger ratio — PCIe or Disk? Explain using bandwidth orders of magnitude.

---

### Q2. (20%) Batch Size and I/O Amortization

With 100% disk offload (Q1 datapoint (5)) fixed, sweep **`--gpu-batch-size = 1, 4, 16`** (each step ×4).

> `--gpu-batch-size N` means FlexGen processes N independent prompts at once. The prompt content doesn't matter for this assignment; only the throughput numbers do.

```shell
python -m flexllmgen.flex_opt --model facebook/opt-1.3b \
    --percent 0 0 100 0 100 0 --gpu-batch-size <1|4|16> \
    --prompt-len 128 --gen-len 16 --offload-dir flexgen_offload \
    2>&1 | tee logs/q2_b<1|4|16>.log
```

> The batch=4 datapoint can be reused from Q1 datapoint (5); you only need to run b=1 and b=16.

<font color="blue">**Questions:**</font>
1. (10%) Tabulate **wall time, throughput, peak gpu mem** for the three batches (1 / 4 / 16).
2. (5%) Compute the throughput ratios between adjacent batches: `b4/b1 ≈ ?`, `b16/b4 ≈ ?`. How do the measured ratios compare to the theoretical 4× (since batch ×4)? **Cross-reference how wall time changes with batch** to explain what you observe.
3. (5%) Looking at how peak GPU mem changes with batch, can batch be scaled indefinitely? Why?

---

### Q3. (20%) Weight Compression and Tier Interaction

`--compress-weight` quantizes weights from FP16 to 4-bit (4× compression); they're de-quantized at load time to feed the GPU. Compare enabling this flag in two scenarios: (α) 100% GPU baseline, (β) 100% Disk offload.

```shell
# (α) 100% GPU + compress
python -m flexllmgen.flex_opt --model facebook/opt-1.3b \
    --percent 100 0 100 0 100 0 --gpu-batch-size 4 \
    --prompt-len 128 --gen-len 16 --offload-dir flexgen_offload \
    --compress-weight 2>&1 | tee logs/q3_alpha.log

# (β) 100% Disk + compress
python -m flexllmgen.flex_opt --model facebook/opt-1.3b \
    --percent 0 0 100 0 100 0 --gpu-batch-size 4 \
    --prompt-len 128 --gen-len 16 --offload-dir flexgen_offload \
    --compress-weight 2>&1 | tee logs/q3_beta.log
```

The non-compressed reference data can be reused from Q1 datapoint (1) and (5).

<font color="blue">**Questions:**</font>
1. (10%) Tabulate the throughput and peak gpu mem under (α) and (β), with and without compression.
2. (10%) Compare the throughput change (direction + magnitude) when `--compress-weight` is enabled in (α) vs (β). From your data, answer:
   - (a) What cost does compression reduce? What cost does it add? Which dominates in each scenario, and how does that determine whether throughput speeds up or slows down?
   - (b) Why is the magnitude of the change (whether speedup or slowdown) not close to the 4× compression ratio?

---

### Q4. (30%) I/O Behavior and Bottleneck Analysis under Disk Offload

This question targets **100% Disk Offload (Q1 dp(5))** and does two things: **Part 1** uses `iostat` to sample disk behavior, **Part 2** uses FlexGen's `--debug-mode breakdown` to measure stage timings.

#### Part 1: Sample disk I/O (two terminals in parallel)

```shell
# Terminal A: Start sampling first
./monitor.sh logs/q4_1_io

# Terminal B: Then run disk offload (same setup as Q1 dp(5))
python -m flexllmgen.flex_opt --model facebook/opt-1.3b \
    --percent 0 0 100 0 100 0 --gpu-batch-size 4 \
    --prompt-len 128 --gen-len 16 --offload-dir flexgen_offload \
    2>&1 | tee logs/q4_1_run.log
# After Terminal B finishes, wait 5–10 seconds for iostat to flush, then Ctrl+C in Terminal A
```

> `iostat` belongs to the `sysstat` package; if not installed: `sudo apt install sysstat`.

Key `iostat -x 1` columns:

- `r/s` — read requests per second
- `rkB/s` — KB read per second
- `rareq-sz` — average size per request (KB); the awk below extracts the **peak** = max value across samples
- `%util` — fraction of time the disk is busy serving requests (saturation)

**Find the disk holding the weight files**:

```shell
lsblk -no PKNAME "$(df --output=source ~/opt_weights/opt-1.3b-np/ | tail -1)"
# e.g. nvme1n1 / sda / vda
```

> On Colab, `lsblk` fails — just use `sda`.

Plug the resulting disk name into the awk below (replace `nvme0n1`) and run it to extract peak values:

```shell
awk '$1=="nvme0n1" {              # ← change to the disk name from above
    if ($2+0 > r) r=$2+0          # r/s peak
    if ($3+0 > k) k=$3+0          # rkB/s peak
    if ($7+0 > sz) sz=$7+0        # rareq-sz peak (max rareq-sz across samples)
    if ($NF+0 > u) u=$NF+0        # %util peak
} END {
    printf "r/s peak       = %s\n", r
    printf "rkB/s peak     = %s\n", k
    printf "rareq-sz peak  = %.2f\n", sz
    printf "%%util peak     = %s\n", u
}' logs/q4_1_io_iostat.log
```

#### Part 2: Run `--debug-mode breakdown` for both scenarios

```shell
# baseline (100% GPU) + breakdown
python -m flexllmgen.flex_opt --model facebook/opt-1.3b \
    --percent 100 0 100 0 100 0 --gpu-batch-size 4 \
    --prompt-len 128 --gen-len 16 --offload-dir flexgen_offload \
    --debug-mode breakdown 2>&1 | tee logs/q4_2_baseline.log

# disk offload (same as dp(5)) + breakdown
python -m flexllmgen.flex_opt --model facebook/opt-1.3b \
    --percent 0 0 100 0 100 0 --gpu-batch-size 4 \
    --prompt-len 128 --gen-len 16 --offload-dir flexgen_offload \
    --debug-mode breakdown 2>&1 | tee logs/q4_2_disk.log
```

> `--debug-mode breakdown` only runs 20 batches to collect timing; **do not compare these throughput numbers to Q1**. We only look at `load_weight` and `compute_layer_decoding` here. The output unit is **seconds**; values across the two scenarios may differ by several orders of magnitude, so keep at least 6 decimal places to avoid misreading.

---

<font color="blue">**Questions:**</font>

**1. (15%) Part 1: disk I/O pattern**

| iostat (disk offload) | Value |
|---|---|
| r/s peak | __ req/s |
| rkB/s peak | __ KB/s |
| rareq-sz peak | __ KB/req |
| %util peak | __ % |

After filling in the table, classify FlexGen's behavior into one of the following and justify your choice using your experimental data:

- **A. Sequential, large blocks**: `rareq-sz peak` ≥ 32 KB (kernel readahead batched consecutive pages into large requests).
- **B. Random, small blocks**: `rareq-sz peak` < 32 KB with high `r/s` (each request independent, kernel can't batch).
- **C. Page cache hit**: `rareq-sz peak` ≈ 4 KB and `rkB/s peak` much lower than your disk's spec (reads don't actually reach the disk).

---

**2. (15%) Part 2: Stage timings in the two scenarios**

| Scenario | load_weight (per-layer, sec) | compute_layer_decoding (per-batch, sec) | load / compute ratio |
|---|---|---|---|
| (1) Baseline | __ s | __ s | __ |
| (5) Disk offload | __ s | __ s | __ |

After filling in the table:
- (a) Compare `compute_layer_decoding` across baseline and disk offload — what do you observe? What does this tell us? (Hint: what does offload change? Is it related to GPU compute volume?)
- (b) Compare `load_weight` across the two scenarios — by how many orders of magnitude do they differ? What aspect of the system does offload change, based on this?

---

## Grading

* Answer all questions in your `.pdf` report, including necessary tables and analysis (hardware info on the first page; see "Submission" below).
* Include the Q4 monitoring logs.

> :mag_right: Each question is graded on the following four levels:
:one: Excellent (100%)
:two: Good (70%)
:three: Poor (30%)
:four: Fail (0%)

:::danger
:warning: Penalty rules:
* Plagiarism — no credit
* Missing hardware info — 10% deduction overall
:::

---

## Submission

#### a. Pack your report and monitoring logs
* Report: `.pdf` only. **First page lists hardware info** (GPU / CPU / RAM / Disk / Path: local or Colab).
* Monitoring logs (Q4 mandatory):
  - `logs/q4_1_io_iostat.log`, `logs/q4_1_io_free.log` (Q4 Part 1 sampling)
  - `logs/q4_2_baseline.log`, `logs/q4_2_disk.log` (Q4 Part 2 breakdown)

```
HW3_<student_id>.zip
    |- HW3_<student_id>/
        |- <student_id>_report.pdf
        |- q4_1_io_iostat.log
        |- q4_1_io_free.log
        |- q4_2_baseline.log
        |- q4_2_disk.log
```

:::danger
Incorrect file format: 20% deduction.
:::

#### b. Submit through the E3 platform before the deadline.

For any questions, please post to the E3 discussion board or email the TA.
