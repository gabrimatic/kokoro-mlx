# Security Policy

## Privacy by Design

Privacy is a core constraint, not a feature toggle.

- **All processing is local.** Inference runs entirely on your Mac via MLX.
- **No network calls** except to download model weights from HuggingFace Hub on first use.
- **No telemetry, no analytics, no cloud.** Zero data leaves your machine after model download.
- **Text stays on device.** Input text is processed locally and never transmitted anywhere.
- **Audio stays on device.** Generated audio is written locally or played back on device.

## Trust Boundaries

| Boundary | Trust Level | Notes |
|----------|-------------|-------|
| User text input | Trusted | Processed locally, never transmitted |
| Generated audio | Trusted | Written to local filesystem or played on device |
| Model weights | Trusted | Downloaded from HuggingFace Hub (Apache 2.0, by hexgrad), cached locally |
| Voice style vectors | Trusted | Bundled with model weights, loaded from local cache |

No remote trust boundaries during inference. The only network access is the initial `huggingface_hub.snapshot_download` call to fetch model weights.

## Vulnerability Reporting

Report vulnerabilities responsibly:

1. **Do not open a public issue.** Vulnerabilities stay private until a fix ships.
2. Use [GitHub's private vulnerability reporting](https://github.com/gabrimatic/kokoro-mlx/security/advisories/new) to submit.
3. Include:
   - Steps to reproduce
   - Demonstrated impact
   - Suggested fix (if any)

Reports without reproduction steps or demonstrated impact are deprioritized.

Expect acknowledgment within 48 hours.

## Out of Scope

These are not considered vulnerabilities:

- Issues requiring physical access to the machine
- Adversarial text inputs designed to produce specific audio outputs
- Mispronunciations or inaccurate prosody

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
