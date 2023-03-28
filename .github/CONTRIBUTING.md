# Contributing

Any contributions are more than welcome.

* Issues: Bug reports, feature requests, and questions.
* Pull Requests: Bug-fixes, feature implementations, and documentation updates.

## Development

Development requires [Poetry](https://python-poetry.org/) as a dependency management system.

```
pip install poetry
```

The `make install` command resolves and installs the dependencies.

```
make install
```

Before making a pull request, run `make format` and make sure `make check` succeeds.

```
make format
make check
```

## Release Procedures For Maintainers

1. make branch for release whose name starts from "release/"
2. Version update pull request is automatically generated. Then, check and accept the pull request.
3. Create pull request from "release/*" to "main" and write release note on the body of the pull request.
4. After the release pull request was merged to main branch, pypi release and release page is automatically uploaded.
    - please do not remove release branch by yourself. Github action will delete it after the procedures are completed.
