# Runtime GPU Benchmark v2

PC2の実運用に近い **B=1** 条件で，State EstimatorとTransitionの推論速度・CUDA VRAMを比較するbenchmarkです．

## v2の変更点

v1ではTransitionをAMP=Trueだけで測定していました．v2では，

- State Estimator：AMP=True / False
- Transition：AMP=True / False × candidate chunk 3 / 6 / 9 / 12 / 18 / 27

を全て比較します．

さらにAMPとFP32で出力がどの程度変わるかも確認し，

- max absolute difference
- mean absolute difference
- RMSE difference
- 0.5 threshold後のbinary disagreement fraction

を`benchmark_results.csv`へ保存します．

Transitionではchunk sizeを変えた場合のFP32出力整合性も確認します．

## 配置

```text
adaptive_control_runtime/
├─ adaptive_runtime_config.yaml
├─ runtime/
├─ server.py
└─ benchmarks/
   └─ benchmark_gpu.py
```

ZIPの`benchmarks/benchmark_gpu.py`を既存の`benchmarks/`へ上書きしてください．

## 実行

```powershell
python benchmarks\benchmark_gpu.py --runtime-config adaptive_runtime_config.yaml
```

通常は追加オプション不要です．

## 自動推奨

各modeについて，

1. Dedicated CUDA VRAMに既定0.75 GB以上の余裕がある設定だけ残す
2. median latency最小を探す
3. FP32が最速設定から3%以内なら，数値的に保守的なFP32を優先する

というルールで，

- `estimator_inference_amp`
- `transition_inference_amp`
- `candidate_chunk_size`

を出力します．

EstimatorとTransitionのmedian latencyを足した`estimated_ml_pipeline_median_ms`も出力します．これは画像後処理・通信・target scoring等を含まないML forward部分の概算です．

## AMPとは

AMP = **Automatic Mixed Precision**です．

通常のFP32（32-bit floating point）だけで計算する代わりに，GPUが安全と判断できる演算をFP16などの低精度で行い，必要な演算はFP32のまま残します．PyTorchでは`torch.autocast(...)`を使っています．

利点：

- VRAM使用量を減らせることがある
- Tensor Coreを利用でき，畳み込み等が高速化することがある

欠点：

- B=1の小さい推論では型変換等のoverheadにより逆に遅くなることがある
- FP32と完全に同じ数値にはならない

checkpoint自体を書き換えるものではありません．AMPをOFFにしても同じ学習済みweightを使います．

今回のclosed-loop制御では，わずかな数値差がcandidate rankingに影響する可能性があるため，速度だけでAMPを選ばず，出力差も確認する設計にしています．

## Windows Shared GPU Memory

PyTorchからTask ManagerのShared GPU Memoryを正確には取得できません．自動推奨はCUDA dedicated-memory統計を使います．

最終設定でruntimeを繰り返し実行し，Task ManagerでShared GPU Memoryがほぼ0 GBのままであることを別途確認してください．

## 出力

```text
benchmarks/results/runtime_gpu_benchmark_YYYYMMDD_HHMMSS/
├─ system.json
├─ benchmark_results.csv
├─ recommendations.json
└─ recommended_runtime_performance.yaml
```
