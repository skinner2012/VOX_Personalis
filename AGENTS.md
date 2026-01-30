# VOX Personalis – Agent instructions

## Commit messages

Always use **Conventional Commits** with optional **Stage-Milestone/Component notation**.

**Format**: `type([SX-MY/]component): subject`

- **SX-MY** (optional): Stage and Milestone (e.g., S1-M0, S2-M3)
- **component**: specific area like specs, api, auth, deps, docs
- **type**: feat, fix, docs, style, refactor, perf, test, chore

**Examples**:
- `feat(S1-M0/specs): add VAD-based silence detection` (with stage/milestone)
- `feat(specs): add VAD-based silence detection` (without stage/milestone)
- `docs(S1-M0/specs): update histogram bucket definitions`
- `fix(api): handle null in user lookup`
- `chore(deps): add webrtcvad and tqdm`

Use imperative mood, keep under ~72 chars. Include Stage-Milestone prefix when work is tied to a specific milestone; omit for general maintenance or cross-cutting changes.
