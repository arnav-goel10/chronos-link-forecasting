# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately by opening a
[GitHub security advisory](https://github.com/arnav-goel10/chronos-link-forecasting/security/advisories/new)
rather than a public issue. Include reproduction steps and the affected commit.

You can expect an acknowledgement within seven days.

## Scope

This is a research repository. It performs no network calls, reads no credentials,
and ships no trained model weights. The checked-in dataset is fully synthetic.

Relevant concerns are therefore limited to correctness and supply chain: dependency
vulnerabilities, and any defect that lets future information reach a training or
evaluation path. Leakage defects are treated as security-relevant here, because a
silently optimistic metric is a correctness failure that misleads readers.
