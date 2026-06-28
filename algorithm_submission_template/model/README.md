# `model/`

Saved model files for inference. `do_save.sh` packs this directory into
`model.tar.gz`; Grand Challenge extracts it to `/opt/ml/model/`.

Expected layout for the current REG2026 integration:

```text
model/
├── config.yaml
├── train_CoT_v01.json                  # or reports.json; used to rebuild Metric A tokenizer
├── metric_a.ckpt                       # or metric_a/metric_a.ckpt
├── metric_b.ckpt                       # or any *.ckpt found under model/
├── pubmedbert/                         # required if Metric B checkpoint uses text_encoder_type=pubmedbert
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
├── conch_v15/
│   └── pytorch_model_vision.bin
├── hest/
│   └── deeplabv3_seg_v4.ckpt
└── grandqc_artifact/
    └── GrandQC_MPP1_state_dict.pth
```

`config.yaml` must use container paths. At minimum, set Metric A checkpoint and
tokenizer corpus paths to files mounted under `/opt/ml/model`, for example:

```yaml
train:
  reports_json_path: /opt/ml/model/train_CoT_v01.json
  model_load_path: /opt/ml/model/metric_a.ckpt
  stage2_model_load_path: /opt/ml/model/metric_a.ckpt
```

Interface 1 also accepts these overrides when present:

```text
METRIC_A_CONFIG_PATH
METRIC_A_CHECKPOINT_PATH
METRIC_A_REPORTS_JSON_PATH
TRIDENT_CONCH_V15_WEIGHTS
TRIDENT_SEG_HEST_WEIGHTS
TRIDENT_SEG_GRANDQC_ARTIFACT_WEIGHTS
```

The HEST and GrandQC artifact checkpoints are required because Interface 1 uses
the same TRIDENT settings as the batch command: `--segmenter hest`,
`--remove_artifacts`, and `--remove_holes`. Runtime has no internet, so these
cannot be downloaded on demand.
