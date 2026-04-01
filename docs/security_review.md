# Security Review

## Threat Model

codebench executes untrusted code (LLM-generated solutions) in sandbox environments. The primary security concerns are:

1. **Sandbox escape** — Generated code breaking out of Docker isolation
2. **Resource exhaustion** — Infinite loops, memory bombs, disk filling
3. **Network access** — Exfiltrating data or attacking external services
4. **Path traversal** — Accessing host filesystem via mount manipulation
5. **Secret exposure** — Leaking API keys through artifacts or logs

## Security Controls

### Sandbox Isolation
- Docker containers with `--network=none` by default
- Memory limits enforced via Docker `--memory` flag (bounded: 64-16384 MB)
- Timeout enforcement via `container.wait(timeout=...)` (bounded: 1-600 seconds)
- PID limit: `--pids-limit=256` prevents fork bombs
- CPU limit: 1 CPU core via `nano_cpus`
- All capabilities dropped: `cap_drop=["ALL"]`
- Read-only root filesystem: `read_only=True`
- Non-root execution: `user="nobody"`
- Workspace mounted read-only

### Path Validation
- All mount paths validated using `os.path.realpath` (resolves symlinks)
- Path traversal blocked in sandbox runners and artifact store
- Artifact store validates all paths stay within base directory via `_safe_resolve()`
- No host paths mounted beyond the designated workspace temp directory

### Secret Handling
- API keys read from environment variables, never stored in config files
- LocalProcessRunner uses minimal environment (PATH, HOME, LANG, TMPDIR only)
- Docker containers do not inherit host environment

### Network Policy
- Benchmark execution is **offline by default**
- Network access requires explicit `network_enabled: true` in config
- Provider API calls are separate from sandbox execution

### Input Validation
- SandboxConfig fields bounded by Pydantic validators:
  - `timeout_seconds`: 1-600
  - `memory_limit_mb`: 64-16384
  - `max_output_bytes`: 1024-100MB

## Remediated Findings

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| 1 | HIGH | Path traversal using normpath instead of realpath | Fixed — uses `os.path.realpath` |
| 3 | HIGH | Docker container missing hardening (PID, CPU, cap_drop) | Fixed — all hardening applied |
| 5 | MEDIUM | Artifact store path traversal | Fixed — `_safe_resolve()` validation |
| 6 | MEDIUM | SandboxConfig unbounded values | Fixed — Pydantic Field constraints |
| 4 | MEDIUM | LocalProcessRunner inherits full host env | Fixed — minimal env only |
| 8 | LOW | Dead `_validate_path` code | Fixed — removed |

## Accepted Risks

| # | Severity | Finding | Rationale |
|---|----------|---------|-----------|
| 2 | HIGH | raw_response persisted unredacted to artifacts | Artifact files are local-only; redacting would prevent legitimate replay/debug. Operators should treat artifact directories as sensitive. |
| 9 | LOW | Provider error bodies may contain echoed credentials | Same as Finding 2. Error bodies are needed for debugging API issues. |
| 7 | LOW | Temp directories orphaned on SIGKILL | `SIGKILL` cannot be intercepted. `finally` blocks handle normal/signal termination. Temp dirs use system default cleanup. |

## Known Limitations

- Local process runner (for testing) does NOT provide isolation — documented as test-only
- Docker-in-Docker scenarios not yet addressed
- No custom seccomp/AppArmor profiles (relies on Docker defaults + cap_drop)
- Docker image field accepts arbitrary values — operators should review configs before running

## Audit Checklist

- [x] Subprocess calls use explicit argument lists (no shell injection)
- [x] Path traversal validated before file operations (realpath + boundary check)
- [x] Docker network disabled by default
- [x] Resource limits enforced (memory, CPU, PID, timeout — all bounded)
- [x] Capabilities dropped in Docker containers
- [x] Sandbox config values bounded by validators
- [x] Artifact store paths validated against base directory
- [x] LocalProcessRunner uses minimal environment
- [x] Timeout enforcement verified
- [ ] No secrets in artifact output (accepted risk — see above)
