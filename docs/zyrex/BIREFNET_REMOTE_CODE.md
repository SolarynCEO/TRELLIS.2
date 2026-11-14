# BiRefNet `trust_remote_code` audit (pinned revisions)

This fork loads BiRefNet via `transformers.AutoModelForImageSegmentation.from_pretrained(..., trust_remote_code=True)`.
To avoid floating remote execution:

1. The model repo is pinned to an explicit commit SHA revision.
2. The downloaded Python files are hash-validated at runtime for known pinned revisions.

## Executed remote modules

At runtime, `transformers` imports modules under:

- `transformers_modules.ZhengPeng7.<repo>.<revision>.BiRefNet_config`
- `transformers_modules.ZhengPeng7.<repo>.<revision>.birefnet`

The class source path is a cached file under:

- `~/.cache/huggingface/modules/transformers_modules/ZhengPeng7/<repo>/<revision>/birefnet.py`

## Pinned revisions and verified file hashes

### `ZhengPeng7/BiRefNet @ e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`

- `BiRefNet_config.py`: `e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52`
- `birefnet.py`: `208771ae626f653d64128fbf2d6ac9f8e645c5cc5e286258a73ec3322bbfe5ef`

### `ZhengPeng7/BiRefNet_dynamic @ 280306042f57b7a33854319da62fd86aaa89ec4c`

- `BiRefNet_config.py`: `e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52`
- `birefnet.py`: `2a45b4e0ece72d7c4212bca1a988e7d7e52bfe9f98ec59c58b8809c8a8b7a831`

### `ZhengPeng7/BiRefNet_HR @ a7a562f6fd16021180f2f4348f4de003a2d3d1e1`

- `BiRefNet_config.py`: `e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52`
- `birefnet.py`: `2a45b4e0ece72d7c4212bca1a988e7d7e52bfe9f98ec59c58b8809c8a8b7a831`

## Why this is not fully vendored

The model architecture code can be vendored, but the checkpoint weights are large and not committed to this repository by policy.  
Given that constraint, revision pinning + file-hash verification prevents unpinned remote code drift in supported variants.
