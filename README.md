# Vectorized $N$-Body Gravitational Simulation & Solar System Dynamics

A Python-based, vectorized $N$-body gravitational simulator utilizing `scipy.integrate.solve_ivp` and `itertools.combinations` to model chaotic and stable astrophysical systems. This project explores fundamental $N$-body problems—from simple binary orbits and chaotic 3-body dynamics (Burrau's Problem) to full 10-body Solar System simulations, barycenter wobbles, planetary resonance spectra, and Martian moon dynamics.

---

## Key Features

- **Vectorized Pairwise Forces**: Efficient computation of $N$-body gravitational interactions using vectorization techniques.
- **High-Precision Integration**: Utilizes ODE solvers (`solve_ivp`) with adaptive step sizes and tunable absolute/relative tolerances to ensure numerical energy conservation ($|\Delta E| < 10^{-5}$ to $10^{-14}$).
- **Spectral & Resonances Analysis**: Fast Fourier Transform (FFT) analysis on orbital dynamics to extract primary orbital frequencies, group envelope periods, and planetary resonances.
- **Ephemeris Data Integration**: Initial condition vectors sourced directly from [JPL Horizons System](https://ssd.jpl.nasa.gov/horizons/app.html#/).

---

## File Architecture & Execution Order

To reproduce the figures and analysis, run the corresponding Python scripts in the order listed below.

| Figure | Script Name | Simulation Subject | Ephemeris / Setup Details | Approx. Runtime |
| :--- | :--- | :--- | :--- | :--- |
| **Figure 1** | `Test case.py` | 2D Stable Binary System | 2 equal-mass bodies ($1\,M_\odot$), circular binary orbit ($T = 4.38\text{ hrs}$) | $< 1\text{ min}$ |
| **Figure 2** | `Burrau's Problem.py` | Pythogoras 3-Body Problem | Masses: $3, 4, 5$. Evolution & escape dynamics over $T=70$ units | $\sim 1\text{ min}$ |
| **Figure 3** | `solar system code.py` | 10-Body Solar System | Target: Sun, Mercury–Pluto<br>Observer: `@500` (Sun center)<br>Epoch: `2024-Nov-29 00:00:00.0000 TDB` | $< 15\text{ mins}$ |
| **Figure 4** | `Sun plot over 70 years.py` | Solar Barycenter Motion | Historical tracking of Solar System barycenter vs. Sun from `1944-Dec-12` | $< 15\text{ mins}$ |
| **Figure 5** | `barycenter plot.py` | Planetary Removal Dynamics | Runs 4 variations (Full System, No Jupiter, No Saturn, No Jupiter/Saturn) over 250 years | $30 - 45\text{ mins}$ |
| **Figure 6** | `Barycenter plot 2.py` | Barycenter Distance & FFT | 500-year evolution and spectral frequency decomposition of barycentric motion | $30 - 45\text{ mins}$ |
| **Figure 7** | `Earth to Mars close encounter.py` | Earth–Mars Distance Spectra | 250-year distance tracking; FFT extracts synodic period ($2.14\text{ yrs}$) and resonance ($15.62\text{ yrs}$) | $< 10\text{ mins}$ |
| **Figure 8** | `phobos and deimos.py` | Martian System Dynamics | Target: Sun, Mercury, Venus, Earth, Mars, Phobos, Deimos, Jupiter, Saturn<br>Epoch: `2024-Nov-29` | $< 15\text{ mins}$ |

> **Note on Bonus Features:**
> - `Test case.py` includes a bonus section for modeling stable star–planet binary dynamics.
> - `Burrau's Problem.py` contains bonus sections with an animated window of system evolution, legacy code versions, and stellar collision handling.

---

## Requirements & Dependencies

The project relies on standard Python scientific libraries along with specific submodules:

```bash
pip install numpy scipy matplotlib
