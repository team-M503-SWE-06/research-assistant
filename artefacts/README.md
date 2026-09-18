# Sample run artefacts

Output of one full live run of the Async Research Assistant, captured
**2026-09-18T19:15:41Z** with `LLM_PROVIDER=anthropic` and
`WEB_SEARCH_PROVIDER=tavily`.

Required by the project brief: "include the output of one full run ... under
`artefacts/` in the repo". Every test in this repository stubs the `ai.*`
package so the suite runs offline with no keys, which means these files are the
only evidence in the repo of the system working against the real Wikipedia,
arXiv, web-search and LLM APIs.

| File | What it shows |
|---|---|
| `demo-run.txt` | all five sample questions answered with `[N]` citations and per-source timings |
| `ask-single.txt` | one question with its reference list |
| `bench.md` | sequential versus concurrent wall-clock time over the five questions |
| `cache-hit.txt` | the same question twice: cold cache, then warm, with `*` marking hits |
| `degraded-run.txt` | one source given an invalid key: the others still answer, and the failure is reported per-source |

Commands, in order:

```bash
python -m researcher demo --no-cache
python -m researcher ask "How does CRISPR-Cas9 gene editing work?" --no-cache
python scripts/bench.py --limit 5
rm -rf ./.cache && python -m researcher ask "How does CRISPR-Cas9 gene editing work?"   # twice, cold then warm
TAVILY_API_KEY=invalid-key-for-degradation-test \
  python -m researcher --log-level INFO ask "How does CRISPR-Cas9 gene editing work?" --no-cache
```

No API keys appear in these files. The script that produced them scans every
artefact for key material and refuses to commit if any is found.
