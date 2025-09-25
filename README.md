# PFTM_Julia
[Julia] Projection Friendly Tree Mapping - improvement on the quantum fermionic boundary operator circuit. Quantum paper for PhD in progress.

# Projection-Friendly Tree Mapping (PFTM)

**Short-depth fermionic mappings for quantum topological algorithms.**

This repository accompanies the paper:

> *Projection-Friendly Tree Mapping: A Log-Depth Fermionic Operator Mapping for Quantum Circuits*  
> (K. Naidoo, 2025, draft in preparation)

It provides:
- Python code (with optional [pytket](https://cqcl.github.io/tket/pytket/api/index.html)) to **construct and measure circuit depths** for the **Projection-Friendly Tree Mapping (PFTM)** and compare against the **Jordan–Wigner (JW)** mapping. This code is an improvement on the quantum circuit first announced in the paper - Representation of the fermionic boundary operator - Akhalwaya et al. - publish in 2022 in Physical Review A - https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.022407
- Reproducible evidence that PFTM reduces worst-case depth from **O(n)** (JW) to **O(log n)**, while preserving bitwise projectors required for topological quantum algorithms (QTDA).

---

## Background

Many quantum algorithms in **Topological Data Analysis (TDA)** and **quantum chemistry** require fermionic operator mappings.  

- **Jordan–Wigner (JW)**  
  - Simple, local definition.  
  - Circuit depth grows **linearly with n** because parity strings are computed serially.

- **Bravyi–Kitaev (BK)**  
  - Logarithmic depth.  
  - But complicates projector structure: weight-\(k\) projections \(P_k\) become expensive.

- **Projection-Friendly Tree Mapping (PFTM)**  
  - Achieves **O(log n)** depth using a binary-tree parity network.  
  - **Preserves bitwise projector semantics**: \(P_k = \sum_{|x|=k} |x\rangle\langle x|\) remains trivial.  
  - Supports **restricted boundary operators** for chain complexes in QTDA.

---

## Repository Structure

```text
├── code/
│   ├── pftm_vs_jw_depth.jl # Julia driver: builds JW/PFTM depth models
│   ├── Manifest.toml       
│   └── Project.toml        
├── README.md               # This file
└── LICENSE


