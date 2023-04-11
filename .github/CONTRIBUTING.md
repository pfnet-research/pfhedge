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

1. Create a release branch named with the prefix "release/".
2. A pull request to update the version information in the release branch is automatically generated. Then, review and merge the pull request.
3. Create a pull request from "release/*" to "main", and write release notes in the description of the pull request.
4. After the release pull request was merged to main branch, pypi release and release page is automatically uploaded.
    - please do not remove release branch by yourself. Github action will delete it after the procedures are completed.

Note: dev/develop branch is no longer used for development. Please make pull request directly into main branch.
