# Modular multi-fidelity simulation framework for advanced sail-driven spacecraft control and dynamics in perturbation-rich regimes (D3_2U)

Isaac Lab environment, drag models and training code for a 2U CubeSat with four deployable D3 booms,
developed for the article:

> Celia Redondo-Verdú, Álvaro Belmonte-Baeza, José L. Ramón, Cristina Cachero, Jorge Pomares,
> **Modular Multi-Fidelity Simulation Framework for Advanced Sail-Driven Spacecraft Control and Dynamics in Perturbation-Rich Regimes**,
> *Results in Engineering*, 2026, 113314, ISSN 2590-1230.
> https://doi.org/10.1016/j.rineng.2026.113314

**This repository complements the article. It is research code shared for transparency and reproducibility,
not a production-ready package**: it is not packaged, tested or maintained as a library, and it depends on
a specific Isaac Lab setup.

## Contents

| Path | Description |
|---|---|
| `d3_2U_env_cfg.py` | Environment configuration (scene, actions, observations, rewards, events, commands) |
| `mdp/` | Isaac Lab MDP terms: boom action, orbit and target commands, perturbation events (drag and gravity gradient), observations, rewards |
| `config/` | Robot configuration |
| `agents/` | RL SKRL agent |
| `collect_dataset/` | Scripts that generate the drag-force dataset with the ray-casting sensor |
| `train_model/` | Notebook that trains the neural-network drag surrogate |
| `orbit_data/` | Pre-computed orbit and atmosphere data |
| `usd/` | Models of the spacecrafts |

## Usage

The imports expect the folder to be located at
`source/isaaclab_tasks/isaaclab_tasks/manager_based/my/D3_2U` inside an Isaac Lab installation.
Commands are run with `./isaaclab.sh -p <script>`, see the docstring of each script for its arguments.

## Citation

```bibtex
@article{RedondoVerdu2026,
  title   = {Modular Multi-Fidelity Simulation Framework for Advanced Sail-Driven Spacecraft Control and Dynamics in Perturbation-Rich Regimes},
  author  = {Redondo-Verd{\'u}, Celia and Belmonte-Baeza, {\'A}lvaro and Ram{\'o}n, Jos{\'e} L. and Cachero, Cristina and Pomares, Jorge},
  journal = {Results in Engineering},
  year    = {2026},
  pages   = {113314},
  issn    = {2590-1230},
  doi     = {10.1016/j.rineng.2026.113314}
}
```

## License

[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).
Files derived from Isaac Lab keep their original BSD-3-Clause notice.
