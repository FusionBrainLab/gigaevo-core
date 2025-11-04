## Development Guidelines

### Git Hooks
Before starting work, you must set up [pre-commit hook](https://pre-commit.com/) for local style checks on your changes.
- Install pre-commit:
```bash
pip install pre-commit
```
- Install git hook scripts:
```bash
pre-commit install
```

### Development

We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for all changes to the project.

#### Commit Message Format

Each commit must follow the format: `<type>[optional scope]: <description>`

**Commit types:**
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code refactoring
- `docs:` — documentation changes
- `test:` — adding or modifying tests
- `chore:` — technical changes (CI/CD, configuration, etc.)
- `perf:` — performance improvements
- `style:` — code formatting without logic changes

**Examples:**
```
feat: add user authentication
fix(loader): handle empty file edge case
```

#### Branch Naming

All branches must follow the format: `<type>/<description>`

Branch types are the same as commit types.

**Examples:**
```
feat/add-authentication
fix/race-condition-in-loader
```

**BREAKING CHANGES:**
If changes break backward compatibility, add `BREAKING CHANGE:` to the commit message body:
```
feat: redesign API interface

BREAKING CHANGE: removed deprecated endpoints /v1/old-api
```

#### Semantic Release

The project uses automated versioning through [semantic-release](https://github.com/semantic-release/semantic-release). The commit message type determines how the version changes:

- `fix:` → **patch** version (1.0.0 → 1.0.1)
- `feat:` → **minor** version (1.0.0 → 1.1.0)
- `BREAKING CHANGE:` → **major** version (1.0.0 → 2.0.0)

**Important:** use correct commit types for versioning to work properly.

#### Pre-merge Checklist

Before creating a PR, ensure that:

- [ ] PR has a clear title
- [ ] Commit messages follow conventional commits format
- [ ] Pre-commit hooks have passed
- [ ] Documentation is updated (if new functionality was added)
- [ ] PR description is complete (what changed, why, test results)

#### Pull Request Guidelines:

1. Each pull request (PR) must have a clear name that accurately reflects the proposed changes. We recommend following the same format as for commits.

2. If you discover a bug or unusual behavior in the code, please create an issue. But before doing so, use the search to ensure your problem hasn't been discussed before.

3. Each PR must include a description of changes: what, why, what problem it solves. If new functionality is added, you must include a usage example.

4. Add tests if possible.

5. Always follow the rule: one PR, one logical change. If the PR title contains 'and' or the PR has grown to 1000+ lines, something likely went wrong.

    **Why this matters**: exceptions exist, but small changes are easier to review and don't take much time to check. Almost no one writes structured and easily readable code all the time. Understanding someone else's flight of fancy is easier when looking at 20-50 lines. When facing 3000 lines, the only thing you want to do is close the tab.

6. All commits must contain meaningful descriptions. This helps colleagues understand the thought process and what was changed.

7. If code will be refined and you don't want reviewers to review it immediately after publishing, add the `[WIP]` (work in progress) tag after `<type>[optional scope]` in the PR title.

8. When publishing a PR, at least one person from the [contributors list](#contributors) must review it, examine the changes, and approve the merge. If a PR is marked as `[WIP]`, it doesn't need to be reviewed.

9. If specific people are needed to review a particular PR, you should add them yourself before first publishing the PR (pressing the `Publish` button).

10. When reviewing a PR, don't publish each comment using the `Comment` button. Instead, use `Draft`. At the end of the review, publish all comments simultaneously by pressing `Publish` at the top of the page. This review method reduces email spam for people subscribed to the project.

11. If a question was asked in an issue, it must be answered. Closing issues with questions is the responsibility of the person who opened them. The reviewer should close the issue or provide feedback within one day after the response.

12. If reviewers don't respond within the timeframes mentioned above, don't hesitate to remind them using pings and direct messages.

## Contributors

-
