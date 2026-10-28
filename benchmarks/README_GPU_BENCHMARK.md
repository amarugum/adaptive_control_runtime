# Runtime GPU Benchmark v3 — four-model FP32

V1.10では4つのruntime modeをすべて対象にします．

- `metadata_only`
- `action_history_only`
- `shock_metadata`
- `shock_action_history`

AMP action-equivalence gateで稀なselected-action差とsafe-count差が確認されたため，本番runtimeは4モデルともFP32固定です．このbenchmarkの目的はAMP再評価ではなく，各EstimatorをGPU上にresidentにした実運用B=1条件で，Transitionの`candidate_chunk_size`を決めることです．

## 実行

`adaptive_control_runtime/`から：

```powershell
python benchmarks\benchmark_gpu.py --runtime-config adaptive_runtime_config.yaml
```

既定ではEstimator/Transitionとも`AMP=False`だけを測定し，各modeについてchunk

`3, 6, 9, 12, 18, 27`

を比較します．

既に確定済みの主比較2モデルについても再計測します．これは4モデルを同じrun・同じPC状態で比較し，設定を1つの`recommendations.json`へまとめるためです．

## 出力

`benchmarks/results/runtime_gpu_benchmark_YYYYMMDD_HHMMSS/`

- `benchmark_results.csv`
- `recommendations.json`
- `recommended_runtime_performance.yaml`
- `system.json`

## Runtime configへ反映

最初にdry-run：

```powershell
python benchmarks\apply_fp32_recommendations.py `
  --runtime-config adaptive_runtime_config.yaml `
  --recommendations benchmarks\results\<RUN>\recommendations.json `
  --dry-run
```

内容が正しければ`--dry-run`を外します．スクリプトは元YAMLのtimestamp backupを作り，各modeへ

```yaml
performance:
  estimator_amp: false
  transition_amp: false
  candidate_chunk_size: <benchmark result>
```

だけを反映します．AMPを自動的に有効化することはありません．

## Shared GPU Memory

PyTorchからWindows Task ManagerのShared GPU Memoryは信頼できる形で取得できません．最終chunk反映後，4モデルsmoke test中にShared GPU Memoryがほぼ0 GBであることを目視確認してください．
