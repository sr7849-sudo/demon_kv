# RiskKV next runs

1. Create one-example smoke data.
2. Submit KVCF GPU smoke.
3. Inspect exact official DynamicKV Mistral launcher.
4. After KVCF smoke passes, generate/submit full KVCF matrix.
5. Run the existing native RiskKV Qwen3 collectors against the same common request set.
6. Run bootstrap/CV statistics only after per-request decisions exist.

The official DynamicKV repository is intentionally kept separate from the KVCF matrix. Its public code supports Mistral-7B-Instruct-v0.2 and model-specific shell launchers; use the exact checked-out launcher rather than inventing a new CLI.
