# This repo is archived

This repo has been made available by Fidelity Investments (FMR LLC) on a read-only basis. We are not currently accepting contributions, and do not anticipate making further updates.

If you have questions, please contact <opensource@fidelity.com>.

## Making your own changes to BoolXAI

Thank you for using BoolXAI! This guide will help you get started and know what to expect when making your own changes.

If you end up using our library in a project, give us a star on GitHub!

### Running the tests

Before beginning we recommend that you ensure that the tests run locally. To do so, you'll need to first install the dev requirements, like so:

```pip install -r requirements_dev.txt```

Then you can run the tests like so:

```pytest .```

Also, make sure that all the notebooks run, and save their output to file (so that they can be read without re-running). Note that this can be time-consuming:

```pytest --nbmake --nbmake-timeout=3600 --overwrite notebooks/*.ipynb```

Important: when updating the notebooks, they must be executed using the `boolxai_test` kernel or else the CI pipeline will fail.

### Checking the test coverage

It's recommended to also check the test coverage for new changes. To do so, run the tests like so (instead of the above):

```coverage run --source=. -m pytest```

Then create an HTML report like so:

```coverage html --omit="notebooks/*.*,setup.py"```

and finally read the report by pointing your browser to `htmlcov/index.html`.

### Building the documentation

To build the HTML documentation from source, using Sphinx, do the following:

```bash
cd docsrc
make html
```

Then point your browser to `docs/index.html` and you should see the web version of the documentation.

Note: when updating docstrings or any other part of the code that might affect the compiled documentation, please make sure to compile the documentation and add the updated `docs/` folder to your PR.
