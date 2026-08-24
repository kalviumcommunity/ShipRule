# Team GitHub Workflow & Conventions

This document outlines the branching model, commit message conventions, pull request guidelines, and issue tracking process for the project.

---

## 1. Branching Strategy

To maintain a clean and stable codebase, our team adheres to the following branching rules:
- **Main Branch (`main`)**: 
  - Holds only production-ready, fully tested, and releasable code.
  - Direct pushes to `main` are strictly prohibited.
- **Feature Branches**:
  - Created for any new features, bug fixes, or documentation tasks.
  - Must follow the naming convention: `[type]/[short-description]` (e.g., `feature/optimal-document-chunking`, `fix/validation-logic`, `docs/api-guide`).
- **Branch Lifecycle**:
  - Feature branches are deleted immediately after they are successfully reviewed, approved, and merged into `main`.

---

## 2. Commit Message Convention

We follow a structured commit message format to maintain a clear Git history and enable automated changelog generation.

### Format
```text
[type]: [short-description]

[optional body explaining why and what changed]
```

### Types
- `feat`: A new feature or capability.
- `fix`: A bug fix.
- `docs`: Documentation changes only.
- `refactor`: A code change that neither fixes a bug nor adds a feature.
- `test`: Adding missing tests or correcting existing tests.
- `chore`: Changes to the build process, auxiliary tools, or libraries (e.g., dependency updates).

### Why
Using this convention ensures that the repository's history is scannable, makes it easy to revert specific sets of features, and allows tools to parse commits automatically for release notes.

---

## 3. Pull Request (PR) Review Process

All code changes must go through a formal pull request review before merging into `main`.

- **Review Approvals**: Every PR requires at least one approval from a peer reviewer before it can be merged.
- **Review Scope**: Code reviews focus on:
  - **Correctness**: Does the code solve the problem without introducing regressions?
  - **Clarity**: Is the code clean, readable, and well-commented?
  - **Data Integrity**: Does the code handle data transitions, storage, and schemas securely?
  - **Test Coverage**: Are unit tests included or updated to verify the changes?
- **Commit History Audit**: Commit messages are reviewed during the PR review process to ensure they conform to the naming conventions.

---

## 4. GitHub Issue Tracking Approach

We use GitHub Issues as our primary tool for planning, organizing, and tracking work.

- **Issue Inception**: Every task, whether a feature request, bug fix, or documentation update, must begin with a registered GitHub Issue.
- **Issue Requirements**: Each issue must contain:
  - An action-oriented, clear title.
  - A comprehensive description outlining the purpose (why it matters) and definition of done (acceptance criteria).
  - An assignee to ensure clear ownership.
  - Relevant labels (e.g., `feature`, `documentation`, `data-pipeline`).
- **Resolution**: Issues are linked to their respective Pull Requests (e.g., using keywords like `Closes #2` in the PR body). The issue is automatically closed when the corresponding PR is merged into `main`.
