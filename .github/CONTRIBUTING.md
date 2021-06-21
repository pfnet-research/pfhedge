# Contributing

Any contributions are more than welcome.

* Issues: Bug reports, feature requests, and questions.
* Pull Requests: Bug-fixes, feature implementations, and documentation updates.

## Development

The `make install` command reads the pyproject.toml file from the current project, resolves the dependencies, and installs them.

```
make install
```

Before making a pull request, make sure `make test` succeeds and run `make format`.

```
make test
make format
```
