## Coding Standards

* **NEVER write new scripts/files without checking existing code first**: Before creating new scripts, ALWAYS search for existing functionality (`Glob` for `**/download*.py`, `**/fetch*.py`, etc.). Extend existing code instead of duplicating. Only create new files when absolutely necessary.
* **Minimalist principle**: Only keep code that is actually necessary. Don't write code that won't be used right now.
* **Follow PFHedge patterns**: When implementing deep hedging, follow established patterns from `examples/snowball_hedge.py`
* **DON'T Apologize in reply**
* **NEVER commit before ask**
* **AVOID excessive comments, unless necessary. Don't add function comments unless being asked**
* **AVOID using `print` for debugging, use `logging` instead**
* **NO docstring in code unless being asked**

### Refactoring Long Functions

When functions become too long (>100 lines), refactor by:
1. **Extract independent code snippets into helper functions**: Move validation, printing, object creation logic into separate functions
2. **Use module-level helpers**: Place helpers outside the class if they don't need instance state (e.g., `_validate_cuda_available()`, `_create_optimizer()`)
3. **Separate concerns**: Split business logic from display logic (e.g., `_print_optimizer_info()`, `_print_training_summary()`)
4. **Keep methods focused**: Main methods should orchestrate, helpers handle details

## Workflow
* TARDIS_API_KEY=TD.rSzoJCymVt13xucv.f0uh1Crgt0pdOcz.jyqhwMa1PoSjzEo.Xwjrec5DEivk4ON.Sgyf7c169jr4RCu.sXMR
* for each task, plan it first and figure out how to test. Then implement it until test pass
* 