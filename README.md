![Pulse Programming Language](pulse_banner_optimized.png)

# 🧠 Pulse Programming Language

[![CI](https://github.com/garet113/pulse-lang/actions/workflows/ci.yml/badge.svg)](https://github.com/garet113/pulse-lang/actions/workflows/ci.yml)
[![GitHub issues](https://img.shields.io/github/issues/garet113/pulse-lang)](https://github.com/garet113/pulse-lang/issues)
[![GitHub stars](https://img.shields.io/github/stars/garet113/pulse-lang)](https://github.com/garet113/pulse-lang/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/garet113/pulse-lang/blob/main/LICENSE)

**Pulse** is a reactive, time-native programming language where variables are alive and update automatically when their dependencies change.  
With built-in time blocks (`@rate`, `@every`, `@after`, `@when`), Pulse unites dataflow, temporal logic, and real-time computation in one minimal language.

---

## ✨ Core Features

- **Reactive by default:** every variable is a live signal.
- **Time primitives:** `@rate`, `@every`, `@after`, and `@when`.
- **Previous state:** use `~x` to reference the last value of `x`.
- **Events:** `emit event(payload)` and `@when event(payload): { ... }`.
- **Tiny runtime:** only ~500 lines of Python.

---

## 📦 Install (local dev)

```bash
python -m pip install -U pip
python -m pip install -e .
pulse example.pulse
```

---

## 🚀 Quick Start

### 1️⃣ Clone
```bash
git clone git@github.com:garet113/pulse-lang.git
cd pulse-lang
```

### 2️⃣ Run the example
```bash
python3 pulse.py example.pulse
```

### 3️⃣ Output
```
comfort: 20.45 temp: 21.0 ΔT: 0.5 led: 255
comfort: 20.60 temp: 21.5 ΔT: 0.5 led: 254
threshold event payload: 20.85
```

---

## 📚 Documentation
- [Language Specification](docs/SPEC.md)
- [Roadmap](docs/ROADMAP.md)

---

## 🧪 Tests

```bash
python3 pulse.py tests/test_reactivity.pulse
```

---

## 🧰 Rust VM (prototype)

The `/vm-rust` folder contains a work-in-progress Rust implementation of the Pulse bytecode VM:

```bash
cd vm-rust
cargo run
```

---

## 🧭 Roadmap

- [x] Reactive interpreter in Python  
- [x] PyPI package + CLI  
- [x] GitHub Actions CI  
- [ ] Rust bytecode VM  
- [ ] Static type inference  
- [ ] WASM backend  
- [ ] Visual timeline IDE (Pulse Studio)

---

## 💬 Share & Support

If you like this project, please ⭐ star it, fork it, or share!  
- Star → [garet113/pulse-lang](https://github.com/garet113/pulse-lang/stargazers)
- Report issues → [Issues](https://github.com/garet113/pulse-lang/issues)

---

## 📜 License

MIT License © 2025 [Garet Johnathan Palmer](https://github.com/garet113)

---

**Website:** https://garet113.github.io/pulse-lang
