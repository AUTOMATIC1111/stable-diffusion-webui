# ADR-004: Upstream Management

## Status
Accepted

## Decision
Record upstream applications in `upstreams.lock.json` with **release tag + full commit SHA**.

```json
{
  "schemaVersion": 1,
  "upstreams": {
    "comfyui": { "repository", "release", "commit", "license", "purpose" },
    "facefusion": { ... }
  }
}
```

## Clone layout
- `runtime/comfyui/` — ComfyUI source at pinned commit
- `runtime/facefusion/` — FaceFusion source at pinned commit

Both directories are **git-ignored**.

## Update process
1. Review GitHub release notes
2. Update `release` and `commit` in lock file
3. Run `python tests/validate-upstreams.py`
4. Run platform `setup-all` script
5. Run `doctor` and `smoke-test`

No automatic major-version upgrades. No `git pull` on default branch during ordinary launch.

## Rollback
Restore previous `upstreams.lock.json` from Git history and re-run setup.

## Security
- HTTPS clone URLs only
- No `curl | bash` installers
- Validate lock file in CI/static tests

## Consequences
- Updates are deliberate and auditable
- Custom nodes and pip deps are not updated unless setup is re-run
