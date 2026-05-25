LLM Offloading and Memory Hierarchy Observation

## Hardware / Environment Information

| Item     | Value                                                                                           |
| -------- | ----------------------------------------------------------------------------------------------- |
| Path     | Google Colab                                                                                    |
| GPU      | NVIDIA Tesla T4                                                                                 |
| VRAM     | 15360 MiB by `nvidia-smi`; 14912 MB by PyTorch                                                  |
| CPU      | Colab x86_64 VM, 2 CPU reported by `iostat`; exact model was not printed in the notebook output |
| Mem      | 12975 MB                                                                                        |
| Disk     | Colab overlay filesystem, 113 GB total, device `sda` for Q4 extraction                          |
| Software | Python 3.12.13, PyTorch 2.10.0+cu128                                                            |

本次實驗使用 Colab Path。後面的分析只針對本次環境的相對差異。

## Q1. Progressive Offload Sweep

### Q1-1 Throughput and Peak GPU Memory

| Datapoint | Weight Distribution | Total Throughput (token/s) | Peak GPU Mem (GB) |
|---|---:|---:|---:|
| (1) | 100% GPU | 81.092 | 2.807 |
| (2) | 50% GPU + 50% CPU | 28.772 | 1.682 |
| (3) | 50% GPU + 50% Disk | 0.721 | 1.713 |
| (4) | 50% CPU + 50% Disk | 0.711 | 0.588 |
| (5) | 100% Disk | 0.357 | 0.620 |
### Q1-2 Boundary Ratios and Bottlenecks

| Boundary | Ratio | Dominant Bottleneck |
|---|---:|---|
| (1) to (2) | 2.82x | PCIe / host-to-device transfer |
| (2) to (3) | 39.91x | Disk |
| (3) to (4) | 1.01x | PCIe effect is almost hidden by Disk |
| (4) to (5) | 1.99x | Disk |
#### (a) Within-class comparison 
先比較兩個 PCIe boundary，也就是 `(1) to (2)` 和 `(3) to (4)`。front-pair 的 PCIe ratio 是 2.82x，但 back-pair 的 PCIe ratio 只有 1.01x。原因是 `(1) to (2)` 發生在還沒有 disk 介入時，把 50% weight 從 GPU 移到 CPU RAM 會讓 decode 多出 CPU RAM 到 GPU 的 PCIe transfer，所以 throughput 明顯下降。相反地，`(3) to (4)` 兩邊都已經有 50% weight 在 disk，disk 已經是 critical path；這時再把另一半 weight 從 GPU 改到 CPU，PCIe 的額外影響會被 disk bottleneck 蓋掉，所以 ratio 幾乎是 1。

再比較兩個 Disk boundary，也就是 `(2) to (3)` 和 `(4) to (5)`。它們差距更大：front-pair 的 Disk ratio 是 39.91x，back-pair 的 Disk ratio 是 1.99x，**所以 front-pair ratio 大很多**。原因是 `(2) to (3)` 是把原本在 CPU RAM 的 50% weight 改成 disk，等於從 DRAM/PCIe 等級掉到 Colab disk 等級，速度差距很大；而 `(4) to (5)` 是在已經有 50% weight 走 disk 的情況下，把 disk-resident weight 從 50% 增加到 100%，比較像 disk traffic 加倍，所以 ratio 接近 2，而不是 40。

#### (b) Cross-class within-pair comparison
在 front pair 裡，PCIe boundary `(1) to (2)` 是 2.82x，Disk boundary `(2) to (3)` 是 39.91x，所以 Disk ratio 比 PCIe ratio 大很多。CPU RAM / PCIe 通常是 GB/s 到十幾 GB/s 等級，而本次 Colab disk 在 Q4 量到的 peak read bandwidth 約 255976 KB/s，也就是約 250 MB/s。Disk 比 PCIe/DRAM 慢數個數量級，所以一旦把 CPU-resident weight 改成 disk-resident weight，throughput 會大幅下降。

在 back pair 裡，PCIe boundary `(3) to (4)` 是 1.01x，Disk boundary `(4) to (5)` 是 1.99x，所以仍然是 Disk ratio 比 PCIe ratio 大。不過這裡兩者差距比 front pair 小很多，因為 back pair 的兩個設定都已經有 disk 在 critical path。`(3) to (4)` 幾乎不變，是因為 disk 已經遮蔽 GPU/CPU placement 的差異；`(4) to (5)` 則仍然會變慢，因為 disk-resident weight 從一半變成全部，disk I/O 量接近加倍。整體來說，**Disk boundary 在兩個 pair 內都比較大**，但 front pair 的差距最大，因為它第一次把資料供給路徑從 CPU/PCIe 拉到 disk。

## Q2. Batch Size and I/O Amortization

### Q2-1 Batch Sweep under 100% Disk Offload

| GPU Batch Size | Wall Time / Total Latency (s) | Total Throughput (token/s) | Peak GPU Mem (GB) |
|---:|---:|---:|---:|
| 1 | 179.478 | 0.089 | 0.489 |
| 4 | 179.418 | 0.357 | 0.620 |
| 16 | 179.345 | 1.427 | 1.051 |

### Q2-2 Throughput Ratios

| Ratio | Measured | Theoretical |
|---|---:|---:|
| b4 / b1 | 4.01x | 4.00x |
| b16 / b4 | 4.00x | 4.00x |

實測 throughput ratio 幾乎等於理論上的 4x。當 batch size 從 1 增加到 4，throughput 從 0.089 增加到 0.357 token/s，因此 b4/b1 = 4.01x。當 batch size 從 4 增加到 16，throughput 從 0.357 增加到 1.427 token/s，因此 b16/b4 = 4.00x。也就是說，在這個實驗中，batch size 每增加 4 倍，總 throughput 幾乎也會增加 4 倍。

這個結果可以從 wall time 的變化解釋。三組實驗的 total latency 幾乎沒有改變：b1 是 179.478 秒，b4 是 179.418 秒，b16 是 179.345 秒。這代表在 batch size 1 到 16 的範圍內，同時處理更多 prompts 並沒有明顯增加總執行時間。原因是 Q2 使用 100% disk offload，主要瓶頸是從 disk 讀取並搬運 model weights，無論 batch 裡有 1、4 或 16 個 prompts，都需要載入同一份 weights。結果就是 wall time 幾乎維持不變，但因為同一段時間內產生更多 output tokens，所以 throughput 接近線性成長。

### Q2-3 Memory Scaling Limit

**不能，batch size 不能無限制放大**。從實驗結果可以看到，peak GPU memory 會隨著 batch size 增加而上升：b1 使用 0.489 GB，b4 使用 0.620 GB，b16 使用 1.051 GB。雖然這個實驗把 model weights 100% offload 到 disk，weights 不會完整常駐在 GPU 上，但 GPU 仍然需要儲存 KV cache、activations、hidden states、temporary tensors 和 runtime buffers。當 batch size 變大時，同時處理的 prompts 變多，這些和 batch 相關的資料結構也會變大。

Peak GPU memory 不會在 batch size 每增加 4 倍時也剛好增加 4 倍，因為總 GPU memory 包含固定成本和 batch-dependent 成本兩部分。固定成本，例如 runtime buffers 和 layer transfer buffers，大致不會隨 batch size 明顯變化；但 KV cache 和 activations 會隨著同時處理的 prompts 數量增加。因此 peak GPU memory 的成長幅度小於 batch size 的成長幅度，但它仍然會持續上升。如果 batch size 繼續增加，最後會碰到 GPU memory 上限，或是讓 GPU compute、CPU memory、disk queueing 變成新的瓶頸。因此，放大 batch size 可以攤平 disk I/O 成本，但只能在有限範圍內有效，不能無限制擴張。

## Q3. Weight Compression and Tier Interaction

### Q3-1 Compression Comparison

| Scenario        | Compression | Total Throughput (token/s) | Peak GPU Mem (GB) |
| --------------- | ----------- | -------------------------: | ----------------: |
| alpha: 100% GPU | No          |                     81.092 |             2.807 |
| alpha: 100% GPU | Yes         |                     23.218 |             1.104 |
| beta: 100% Disk | No          |                      0.357 |             0.620 |
| beta: 100% Disk | Yes         |                      1.732 |             0.470 |

### Q3-2 Interpretation

#### (a) Compression 減少什麼成本、增加什麼成本？哪個成本主導？

`--compress-weight` 減少的是 **weight storage / transfer cost**。因為 weight 從 FP16 被量化成 4-bit，weight bytes 約變成原本的四分之一，所以需要從 GPU memory、CPU memory 或 disk 搬運的 weight 資料量會下降。同時，compression 也會增加 **de-quantization / unpacking cost**，因為被壓縮的 weight 在實際餵給 GPU 計算前，需要先被解壓或轉回可計算的格式。

在 alpha，也就是 100% GPU baseline，compression 讓 throughput 從 81.092 降到 23.218 token/s，約為原本的 0.286x。這代表 alpha 啟用 compression 後是變慢的。原因是 alpha 的 weights 原本就放在 GPU 上，weight 供給路徑已經很快，主要瓶頸不是 disk I/O 或資料搬運。因此 compression 雖然減少 weight footprint，卻沒有明顯縮短主要瓶頸；反而新增的 de-quantization / unpacking 成本變成主導因素，所以 throughput 下降。

在 beta，也就是 100% disk offload，compression 讓 throughput 從 0.357 增加到 1.732 token/s，約加速 4.85x。這代表 beta 啟用 compression 後是變快的。原因是 beta 的主要瓶頸是從 disk 讀取 model weights，disk I/O 很慢；compression 讓需要從 disk 讀出的 weight bytes 大幅減少，所以最慢的 disk transfer 成本被降低。雖然 de-quantization / unpacking 仍然是額外成本，但在 beta 中，減少 disk I/O 的收益大於新增解壓成本，因此整體 throughput 上升。

#### (b) 為什麼 throughput 變化幅度不會接近 4x compression ratio？

因為 4x compression ratio 只代表 **weight bytes 約減少 4 倍**，不代表整個 inference 的 end-to-end latency 也會剛好減少 4 倍。一次推論的總時間不只包含 weight transfer，還包含 prefill、decode compute、kernel launch、KV cache 讀寫、activation / hidden state 處理、framework overhead，以及 compression 帶來的 de-quantization / unpacking 成本。只有 weight transfer 這一部分直接受 4-bit compression 影響。

因此，在 alpha 中，原本主要瓶頸不是 weight transfer，而是 GPU computation 加上 compression 後的解壓成本；所以 weight 變小 4 倍不會帶來 4x speedup，反而因為新增解壓成本造成 3.49x slowdown。在 beta 中，weight transfer 確實是主要瓶頸，所以 compression 可以明顯加速，實測 speedup 是 4.85x；但這個數字仍然不是單純由 4x compression ratio 決定，而是整個 pipeline 中 disk I/O 減少、解壓成本增加、cache 行為、系統 I/O 排程和其他固定開銷共同作用的結果。

## Q4. I/O Behavior and Bottleneck Analysis under Disk Offload
### Q4-1 Disk I/O Pattern

| iostat (disk offload, `sda`) | Value |
|---|---:|
| r/s peak | 2583 req/s |
| rkB/s peak | 255976 KB/s |
| rareq-sz peak | 128.00 KB/req |
| %util peak | 80.8% |

根據實驗數據，FlexGen 在 100% disk offload 下的 I/O pattern 應分類為 **A. Sequential, large blocks**。最直接的依據是 `rareq-sz peak = 128.00 KB/req`，大於題目給的 32 KB 門檻。`rareq-sz` 代表每一次送到 block device 的 read request 平均有多大，合理的解釋是，FlexGen 在連續讀取 weight file，而 kernel readahead 或 block layer merging 將相鄰的 pages 合併成較大的 read requests。

換句話說，FlexGen 在 disk offload 時比較像是把 weight 當成一段連續的資料流讀進來，而不是一直跳到不同位置讀小區塊。這就是為什麼它符合 **sequential large-block streaming** 的特徵。

這個分類也可以從 bandwidth 和 utilization 解釋。`rkB/s peak = 255976 KB/s`，大約是 250 MB/s，代表真的有大量資料從 disk device 被讀出來；`%util peak = 80.8%`，表示 disk 在高峰期間大部分時間都很忙。這兩個數字說明資料並不是大多從 page cache 直接命中，因為如果是 page cache hit，進到 disk device 的 read bandwidth 和 utilization 應該會低很多。它也不像 random small-block read，因為 random small-block pattern 通常會看到 `rareq-sz` 小於 32 KB。

因此，Q4-1 驗證：FlexGen 在 100% disk offload 時，確實會對存放 weights 的 disk 產生大量 sequential large-block reads。也驗證了前面 Q1/Q2 看到 disk offload throughput 很低，懷疑瓶頸是 disk I/O；Q4-1 則用 `iostat` 證明 disk path 真的有大量 weight streaming traffic，而且 disk device 本身有明顯負載。所以低 throughput 可以合理歸咎為 disk I/O supply 成為 bottleneck，而不是 GPU compute 本身突然變慢。

### Q4-2 Stage Timing

| Scenario | load_weight (per-layer, sec) | compute_layer_decoding (per-batch, sec) | load / compute ratio |
|---|---:|---:|---:|
| (1) Baseline | 0.000036 | 0.006474 | 0.0056 |
| (5) Disk offload | 0.225546 | 0.006704 | 33.64 |

Q4-2 的把 FlexGen 的執行時間拆成不同 stage，確認 100% disk offload 主要放大哪一段成本。這裡比較的是 100% GPU baseline 和 100% disk offload；兩個實驗使用同一個 OPT-1.3B、相同 batch size、相同 prompt length 和 generation length，所以理論上 GPU 要做的 decoding computation volume 應該接近。因此可以分析 offload 對 weight supply path 的影響。

#### (a) Compare `compute_layer_decoding` across baseline and disk offload

先看 `compute_layer_decoding`。Baseline 是 0.006474 秒，disk offload 是 0.006704 秒，兩者只差約 3.6%，幾乎在同一個量級。這代表 disk offload 並沒有明顯增加每一層在 GPU 上實際做 decoding computation 的時間。模型大小、batch size 和 generation workload 都沒有改變，因此 compute stage 維持接近是合理的。

這個觀察讓我們可以把 throughput 下降的原因往資料供應路徑看。也就是說，offload 主要改變的是每層計算前 weights 要從哪一層 memory hierarchy 被送到 GPU。

#### (b) Compare `load_weight` across baseline and disk offload

接著看 `load_weight`，差異就非常明顯。Baseline 的 `load_weight` 只有 0.000036 秒，因為 weights 已經在 GPU local memory 裡，每層要使用 weight 時幾乎可以直接取得。Disk offload 則變成 0.225546 秒，代表每層計算前都需要等待 weights 從 disk 供應上來。兩者相差約 6265 倍。

`load / compute ratio` 也把 critical path 的轉移表現得很清楚。Baseline 的 ratio 只有 0.0056，表示 load_weight 在整體 stage timing 中幾乎可以忽略，主要成本仍然落在 compute。Disk offload 的 ratio 則是 33.64，表示 load_weight 是 compute_layer_decoding 的 33 倍以上。這等於說 GPU compute stage 本身很快，但整個 pipeline 的進度被 weight loading 牽住。

因此，Q4-2 驗證了：100% disk offload 會把 critical path 從 GPU compute 推向 weight loading / data movement。這和 Q4-1 的 `iostat` 結果互相對應：Q4-1 看到 disk 有大量 sequential large-block reads，Q4-2 則看到 `load_weight` 大幅上升而 `compute_layer_decoding` 幾乎不變。兩者合起來說明，disk offload 下真正主導 latency 的是 disk I/O supply。
