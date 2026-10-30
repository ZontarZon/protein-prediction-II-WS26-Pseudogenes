# Commit Convention

Group related work into **one commit**. A feature or fix and the docs, refactor, or chores that belong to it stay together. Do not split those by type.

Split only when the changes do not belong together: a fix and a feature, or code that is otherwise unrelated.

## Message

Subject: `[type] brief summary`

- `type` is the **most important** part of the commit. If the commit adds a feature and also updates docs, the subject is `[feat]`, not `[docs]`.
- Use `[fix]` the same way when a bug fix is the main change.
- Use `[docs]`, `[refactor]`, or `[chore]` only when that is the whole commit.

Body: one line per kind of work included, tagged with its type.

```
[feat] classify UniProt pseudogene candidates

feat: filter Swiss-Prot entries that carry the pseudogene caution
docs: describe the candidate set in the README
chore: ignore downloaded embedding files
```

## Types

| Type         | Use for                                                |
| ------------ | ------------------------------------------------------ |
| `[feat]`     | New features (new code, experiments, script logic)     |
| `[fix]`      | Bug fixes                                              |
| `[docs]`     | Documentation (README, reports)                        |
| `[refactor]` | Code restructuring that is neither a feature nor a fix |
| `[chore]`    | Housekeeping (`.gitignore`, job scripts, build config) |

## When to split

Split:

- a bug fix and a new feature
- changes that are not part of the same piece of work

Keep in one commit:

- a feature and the README or report that documents it
- a feature and the job script or ignore rules it needs
- a fix and the small cleanup required to make that fix
