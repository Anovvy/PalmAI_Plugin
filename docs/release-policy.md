# PalmAI Release Policy & Compatibility Matrix

This document defines the versioning guidelines, compatibility contracts, and security boundaries governing PalmAI releases.

---

## 1. Versioning Specification

PalmAI follows [Semantic Versioning (SemVer 2.0.0)](https://semver.org/):

$$\text{v}\langle\text{MAJOR}\rangle.\langle\text{MINOR}\rangle.\langle\text{PATCH}\rangle$$

- **MAJOR (v2.0.0):** Breaking protocol changes, incompatible input/output GIS formats, or QGIS major architecture migrations (e.g., future Qt6 / QGIS 4 support).
- **MINOR (v1.1.0):** New analysis tools, upgraded model architectures, new backend features, or backward-compatible schema enhancements.
- **PATCH (v1.0.1):** Bug fixes, performance tuning, path mapping updates, or dependency security patches.

---

## 2. Component Compatibility Matrix

| PalmAI Plugin Version | Worker Protocol Version | Container Image Tag / Digest | Model Weights Release | Supported QGIS Versions | Supported Python (Venv) |
|---|:---:|---|:---:|:---:|:---:|
| `2.0.0` | `1.0` | `ghcr.io/anovvy/palmai-core:v2.0.0-cpu`<br>`ghcr.io/anovvy/palmai-core:v2.0.0-gpu` | `models-v1` | QGIS 3.22 LTR – 3.99 | Python 3.9 – 3.12 |
| `1.0.0 (Legacy)` | `1.0` | `ghcr.io/anovvy/palmai-core:v1.0.0-cpu` / `:cpu`<br>`ghcr.io/anovvy/palmai-core:v1.0.0-gpu` / `:gpu` | `models-v1` | QGIS 3.22 LTR – 3.99 | Python 3.9 – 3.12 |

> **Note on QGIS 4:** QGIS 4.x introduces major breaking architectural changes (Qt6/PyQt6 migration, Python API redesign). PalmAI explicitly targets QGIS 3.x (`qgisMaximumVersion=3.99`). QGIS 4 support is planned for a future major release.

---

## 3. Cryptographic Model Weight Registry

PalmAI model checkpoints are hosted as GitHub Release assets under release tag `models-v1`. Both newly downloaded and locally cached weights are verified against the following authoritative SHA-256 checksums before loading:

| Filename | Task / Model | Size | SHA-256 Checksum |
|---|---|---|---|
| `yolov8n_counting_1.0.pt` | Palm Oil Tree Counting (YOLOv8) | ~6.3 MB | `2c5a4bec86354e808cbd3324fe36656c1a3e1be342ddc1797087e9c48592cfd8` |
| `yolov8n_centertree_1.3.pt` | Palm Oil Centre Tree (YOLOv8) | ~6.4 MB | `6ed11bab0e8fd3b61a7ab5d751b76d6096d958ce02d0870919667b4048ceef35` |
| `yolov8n_classification_2.1.pt` | Condition Classification (YOLOv8) | ~6.3 MB | `05162683b0842e759d17e3448331ece0b5c12678c8d3c66f5d296e1d9b02f795` |
| `deeplabv3+_roaddetection_1.pth.tar` | Road Network Detection (DeepLabV3+) | ~504 MB | `6f8ec47d0d5be7573f128bb6f1ec53b6dd185fd90bcec2975c5efc451cea9ee0` |

### Security Policy
1. **Fail Closed:** Any checkpoint whose SHA-256 hash does not match the authoritative registry is rejected immediately and deleted to prevent using corrupted or tampered files.
2. **Atomic Promotion:** Weights are downloaded to a `.part` temporary file and promoted via atomic replacement (`os.replace`) only after verifying the SHA-256 digest.
3. **Execution Trust Boundary:** PyTorch checkpoint deserialization (`torch.load`) is executed only on weights verified against the cryptographic registry above.

---

## 4. Container Tag & Rollback Policy

### Tagging Standards
- **Versioned Tags (`vX.Y.Z-cpu`, `vX.Y.Z-gpu`):** Immutable release identifiers. Once published to GHCR, versioned tags are never overwritten.
- **Compatibility Aliases (`:cpu`, `:gpu`):** Mutable floating tags maintained for backward compatibility with earlier plugin versions.

### Rollback Procedure
If a regression is identified in a container update:
1. Identify the previous stable image digest in the GitHub Actions `publish-images` summary or release notes (e.g., `sha256:abcd...`).
2. Pull the known-good image digest directly:
   ```bash
   docker pull ghcr.io/anovvy/palmai-core@sha256:<previous-stable-digest>
   ```
3. Retag locally to the required versioned tag:
   ```bash
   docker tag ghcr.io/anovvy/palmai-core@sha256:<digest> ghcr.io/anovvy/palmai-core:v1.0.0-cpu
   ```
