# Documentation developer documentation


## Executing and building examples.

Here is the idea. Goal: If I would download pymob, I can ensure that all the examples are running without problems if I also install the latest version of the example. This requires that the main branch of the examples are always up to date with the main branch of pymob. 

This goal brings the obvious benefit that it makes pymob user friendly, but more importantly. **Achieving this means to test pymob on every release with complex examples.** This is bound to catch more errors and make the software more robust.

Also the documentation examples ***must be based on code that is actually executed***. Why? Because it will help remove obsolete code examples. Using pymob to write the documentation is also more fun and will help make the software more user friendly.

### Testing docstring examples with doctest

Testing a single module
```bash
python -m doctest --fail-fast  pymob/simulation.py 
```

Using pytest to run multiple modules
```bash
# pytest needs to be run directly inside the module where the code is located
cd pymob 
pytest --doctest-modules --disable-warnings \
    --ignore=inference/interactive.py \
    --ignore=inference/sbi \
    --ignore=inference/optimization.py
cd ..
```


### Using jupyter notebooks for the documentation

Execute a notebook and convert it to markdown using mbconvert with the `--execute` flag.

```bash
jupyter nbconvert --to markdown --execute docs/source/user_guide/*.ipynb
```

### Using jupyter notebooks for examples in the documentation

Testing should be done on the case study level. Once testing passes it is relatively safe to assume that the notebooks also work. Then, we can follow a routine like such:

PROPOSAL:

1. Checkout case study 
2. Use jupyter nbconvert to execute the notebook and convert it to the docs/source/examples folder as markdown

This can become a complete pipeline job for each case study. The command could look like this. Jobs can run in parallel

```bash
# checkout
git clone git@github.com:flo-schu/CASE_STUDY case_studies/CASE_STUDY

# run tests write a case study testing bash script that can be used on the cluster
cd case_studies/CASE_STUDY 
pytest --fail-fast
cd ../..

# execute notebook convert and store in docs directiory this step depends on successful test run
jupyter nbconvert --to markdown --execute --output_dir docs/source/examples/CASE_STUDY case_studies/CASE_STUDY/scripts/example_1.ipynb

# repeat last step for 2nd example, ...
```


These commands should be integrated in pre-release CI pipelines. This is because more sophisticated notebooks, will take quite some time to compile. This is usually unnecessary when making development releases or pre-releases. But when updating the standard release available at `pip install pymob` (e.g. 0.4.1), then the examples in the documentation 

## Compile documentation

Finally the documentation is compiled for testing. This should be 

```bash
sphinx-apidoc  -o docs/source/api pymob && sphinx-build -M html docs/source/ docs/build/
```



