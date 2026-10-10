# Agent hooks

## Claude Code

Add to `~/.claude/settings.json` (user) or `.claude/settings.json` (project):

```json
{
  "hooks": {
    "Stop": [
      {"hooks": [{"type": "command", "command": "SOURCEMARK_MODE=warn sourcemark hook stop", "timeout": 30}]}
    ]
  }
}
```

When the agent finishes a turn, the hook reads the session transcript the runtime hands it,
rebuilds what the agent read and wrote (Read, Write, Edit, Grep, `cat`/`sed -n`/`head`, heredoc
writes, URLs in any tool output), and checks every citation in that turn.

| `SOURCEMARK_MODE` | Behaviour |
|---|---|
| `shadow` (default) | Record the check in the ledger; say nothing |
| `warn` | Show a one-line summary to the user |
| `enforce` | If any citation is unsupported, send the list back to the agent to fix or remove (a `block` decision). A second stop in the same turn only warns, so it never loops |
| `off` | Do nothing |

The hook fails open: a parse or ledger error never blocks the agent. It re-reads the transcript
briefly if the final reply has not been flushed when the hook starts.

Subagent transcripts stored beside the session (`<session>/subagents/*.jsonl`) are loaded as
*delegated* evidence: citations only a subagent read are reported as `delegated`, not as failures.

## Other runtimes

`sourcemark check` accepts Claude Code's JSONL format. For other runtimes, build a `Session`
from tool results with `sourcemark.observe.observe_tool(name, input, structured_result, content, cwd)`
and call `sourcemark.check.check_text(answer, session)`.

<!-- agent-provenance
agent: claude-code:studio:07349816-9a65-42c5-b724-2275f752918f
node: studio
written: 2026-10-07
reasoning: agent-trace 07349816-9a65-42c5-b724-2275f752918f
-->
