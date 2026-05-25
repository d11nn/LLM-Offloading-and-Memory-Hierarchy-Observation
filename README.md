# LLM Offloading and Memory Hierarchy Observation

This project studies how FlexGen behaves when model weights move across GPU, CPU RAM, and disk on a Colab T4 environment.

## What the report shows

- **Baseline is fastest:** keeping weights on GPU gives the highest throughput and the largest GPU memory footprint.
- **CPU offload is moderate, disk offload is severe:** moving weights from GPU to CPU slows things down, but moving them to disk causes the biggest drop by far.
- **Batching helps throughput:** with 100% disk offload, larger batch sizes keep wall time nearly flat while throughput rises almost linearly.
- **Compression is workload-dependent:** `--compress-weight` hurts the GPU-resident case, but helps the disk-offload case because it reduces transfer volume more than it adds decompression cost.
- **Disk offload is I/O-bound:** `iostat` shows large sequential reads, and `load_weight` dominates `compute_layer_decoding` under disk offload.

## Environment

The experiments were run in Google Colab on an NVIDIA Tesla T4 with the Colab overlay filesystem as the disk tier.

## Full report

See `314581047_report.md` for the complete tables, measurements, and detailed analysis.
