# Contributing to UltraEP

Thank you for improving UltraEP. Keep changes focused and include the test or
benchmark evidence needed to evaluate them.

## Development workflow

1. Create a topic branch from the current development branch.
2. Make the smallest coherent change and update user-facing documentation.
3. Run the CPU unit suite with `python -m pytest`.
4. For runtime changes, run the relevant GPU integration test. For HCU or
   transport changes, include the machine topology and launcher command in the
   pull request.
5. Run `bash format.sh` before submitting.

See [tests/README.md](tests/README.md) for the test taxonomy and commands, and
[docs/hcu.md](docs/hcu.md) for HCU-specific validation.

## Pull requests

Describe the motivation, behavior change, compatibility impact, and validation
performed. Do not commit generated wheels, build trees, traces, benchmark logs,
cluster addresses, credentials, or machine-specific topology files.

Performance claims should include hardware, software versions, world size,
model/test shape, warmup count, measurement count, and the baseline command.
