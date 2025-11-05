# 🧠 Pulse Programming Language

**Pulse** is a new experimental **time-aware, reactive-by-default** programming language.  
Instead of waiting for inputs, Pulse variables *live and breathe* — they automatically update when their dependencies change.

> 🪄 Never-before-combined design: reactive dataflow + built-in time semantics.

---

## ✨ Core Features

- **Reactive by default:** every variable is a live signal.
- **Time primitives:** `@rate`, `@every`, `@after`, and `@when`.
- **Previous state:** use `~x` to reference the last value of `x`.
- **Events:** `emit event(payload)` and `@when event(payload): { ... }`.
- **Tiny runtime:** only 500 lines of Python.

---

## 🚀 Quick Start

### 1️⃣ Clone
python3 pulse.py example.pulse

comfort: 20.45 temp: 21.0 ΔT: 0.5 led: 255
comfort: 20.60 temp: 21.5 ΔT: 0.5 led: 254
threshold event payload: 20.85

```bash
git clone https://github.com/garet113/pulse-lang.git
cd pulse-lang
python3 pulse.py example.pulse
# -Pulse-Programming-Language
Pulse is a reactive, time-native programming language where variables are alive and update automatically when their dependencies change. With built-in time blocks (@rate, @every, @after, @when), Pulse blends dataflow, temporal logic, and real-time computation in under 500 lines of Python.

example of code
const TWO = 2
let temp = sensor("temp")
let hum  = sensor("humidity")
let comfort = smooth((temp * 0.7) + (hum * 0.3), 3)
let delta_temp = temp - ~temp
let wave = sin(now() * TWO * 3.14159)
let led  = clamp((wave + 1) * 128, 0, 255)

@rate 2Hz: {
  print("comfort:", comfort, "temp:", temp, "ΔT:", delta_temp, "led:", led)
}

@every 1s: {
  temp = temp + 0.5
}

@after 5s: {
  emit threshold(comfort)
}

@when threshold(payload): {
  print("threshold event payload:", payload)
}
