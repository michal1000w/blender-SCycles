# Diffraction albedo acceptance check

From the repository root, run:

```sh
python3 tests/performance/run_cycles_diffraction_albedo_acceptance.py
```

The runner compiles only `cycles_diffraction_albedo_acceptance_test.cpp` with
the selected `CXX` compiler (default `c++`), Cycles' namespace definitions,
and the first bundled TBB include directory under `lib/`. It then runs the
CPU executable. The executable and `results.json` are written to
`build/tests/performance/cycles_diffraction_albedo_acceptance/`. Set `CXX` to
a shell-quoted compiler command to pass compiler options, for example
`CXX='clang++ --stdlib=libc++'`. Use `--output-dir DIR` to select another
output directory or `--results-json FILE` to put the JSON elsewhere.

The runner returns the compiler's exit code on build failure, or the test's
exit code after a successful build. It writes JSON in either case, including
the source hash, compiler command and version, exact compile argument array,
executable path and hash when available, and captured build/test output.
