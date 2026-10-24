# Toy CoVe fixture: Mexican–American War dates

Demonstrates CoVe correcting a **wrong end year** in the baseline (1849 → 1848).

| File | Role |
|------|------|
| `manifest.json` | `query_id`, user query text |
| `pipeline.json` | Fixture outputs for all four CoVe steps |

Invoke via `POST /verify` with `"query_id": "mexican_american_war"` or `POST /eval/bootstrap-toy`.
