# PFTM_Julia
[Julia] Projection Friendly Tree Mapping - improvement on the quantum fermionic boundary operator circuit. Quantum paper for PhD in progress.

## 📊 Traffic & Popularity
![Clones](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FPFTM_Julia-clones.json%3Fv%3D5)
![Forks](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FPFTM_Julia-forks.json%3Fv%3D1)
![Views](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F9b749f24de62343dc995f8d524027c39%2Fraw%2FPFTM_Julia-views.json%3Fv%3D5)
![Stars](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FPFTM_Julia-stars.json%3Fv%3D5)
![Commits](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FPFTM_Julia-commits.json%3Fv%3D5)

> Auto-updated daily at 14:00 UTC via GitHub Actions.

## 📈 Metrics
![Activity (last 4 weeks)](https://raw.githubusercontent.com/kugendran-naidoo/PFTM_Julia/main/metrics/activity_4w.png)

> Auto-updated daily at 14:00 UTC via GitHub Actions.

# Projection-Friendly Tree Mapping (PFTM)

**Short-depth fermionic mappings for quantum topological algorithms.**

This repository accompanies the paper:

> *Projection-Friendly Tree Mapping: A Log-Depth Fermionic Operator Mapping for Quantum Circuits*  
> (K. Naidoo, 2025, draft in preparation)

It provides:
- Julia code to **construct and measure circuit depths** for the **Projection-Friendly Tree Mapping (PFTM)** and compare against the **Jordan–Wigner (JW)** mapping. This code is an improvement on the quantum circuit first announced in the paper - Representation of the fermionic boundary operator - Akhalwaya et al. - publish in 2022 in Physical Review A - https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.022407
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


