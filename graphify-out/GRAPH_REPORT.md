# Graph Report - minisam3d  (2026-10-10)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 7 nodes · 2 edges · 5 communities (1 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 556 input · 63 output

## Graph Freshness
- Built from commit: `9f15bd85`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Sophos Authentication Service
- Radxa Dragon Q6A Hardware
- Model Quantization
- ONNX Runtime QNN Integration
- Qualcomm AI Platform

## God Nodes (most connected - your core abstractions)
1. `Sophos Client Authentication Agent` - 2 edges
2. `qemu-x86_64-static` - 1 edges
3. `sophos-caa.service` - 1 edges
4. `Radxa Dragon Q6A` - 0 edges
5. `W8A16 Quantization` - 0 edges
6. `ONNX Runtime QNN EP` - 0 edges
7. `Qualcomm AI Hub` - 0 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Radxa Network Authentication** — sophos_caa, qemu_user_static, systemd_sophos_service [EXTRACTED 1.00]

## Communities (5 total, 4 thin omitted)

### Community 0 - "Sophos Authentication Service"
Cohesion: 0.67
Nodes (3): qemu-x86_64-static, Sophos Client Authentication Agent, sophos-caa.service

## Knowledge Gaps
- **6 isolated node(s):** `qemu-x86_64-static`, `sophos-caa.service`, `Radxa Dragon Q6A`, `W8A16 Quantization`, `ONNX Runtime QNN EP` (+1 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 6 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `qemu-x86_64-static`, `sophos-caa.service`, `Radxa Dragon Q6A` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._