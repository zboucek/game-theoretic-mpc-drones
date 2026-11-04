# Licensing and third-party dependencies

The original framework, tests, utilities and written documentation are licensed under the [MIT License](LICENSE). Data and figure files in `data/` and `docs/figures/` are licensed under [CC BY 4.0](LICENSE-DATA). This source distribution contains our wrapper/controller code and a simulator wheel. Other dependencies are installed separately. Installed dependencies retain their own licenses; the MIT license does not replace those licenses.

| Dependency | Reproduction pin | License / upstream |
|---|---|---|
| NashOpt | 1.1.0 | [Apache-2.0](https://github.com/bemporad/nashopt/blob/main/LICENSE) |
| gym-pybullet-drones | `01563120e060c9233a61f9cf00e6d4e5a8b331ab` (2.2.0+mpc.01563120) | [MIT](https://github.com/learnsyslab/gym-pybullet-drones/blob/01563120e060c9233a61f9cf00e6d4e5a8b331ab/LICENSE), copyright Jacopo Panerati |
| NumPy | 2.4.2 | [BSD-3-Clause](https://github.com/numpy/numpy/blob/v2.4.2/LICENSE.txt) |
| SciPy | 1.17.1 | [BSD-3-Clause](https://github.com/scipy/scipy/blob/v1.17.1/LICENSE.txt) |
| JAX / jaxlib | 0.7.2 | [Apache-2.0](https://github.com/jax-ml/jax/blob/main/LICENSE) |
| JAXopt | 0.8.5 | [Apache-2.0](https://github.com/google/jaxopt/blob/main/LICENSE) |
| HiGHS / highspy | 1.13.1 | [MIT](https://github.com/ERGO-Code/HiGHS/blob/master/LICENSE.txt) |
| OSQP | 1.1.1 | [Apache-2.0](https://github.com/osqp/osqp/blob/master/LICENSE) |
| Matplotlib | 3.10.8 | [Matplotlib license (PSF-based)](https://matplotlib.org/stable/project/license.html) |
| PyBullet | 3.2.7 | [zlib](https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt) |
| Gymnasium | 1.2.3 | [MIT](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE) |
| transforms3d | 0.4.2 | [BSD-2-Clause](https://github.com/matthew-brett/transforms3d/blob/main/LICENSE) |
| python-control | 0.10.2 | [BSD-3-Clause](https://github.com/python-control/python-control/blob/main/LICENSE) |
| Stable-Baselines3 | 2.7.1 | [MIT](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE) |
| PyTorch | 2.10.0 CPU | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) |

NashOpt 1.1.0 is an external Apache-2.0 dependency, installed separately from this MIT-licensed framework. No NashOpt solver sources are modified or redistributed in this release. Calling its public API does not relicense NashOpt or impose Apache-2.0 on the original wrapper code; separable works that merely bind to its interfaces are excluded from the license's definition of derivative works. Apache-2.0 permits both commercial and noncommercial use; noncommercial use is not an exemption from its conditions. See [Apache sections 1 and 4](https://www.apache.org/licenses/LICENSE-2.0).

If you redistribute NashOpt itself, include its Apache-2.0 license and preserve the applicable copyright and attribution notices, including upstream NOTICE content if supplied. If you modify and redistribute its files, mark the changes as required by section 4. Other dependencies have their own redistribution conditions; a citation or link alone does not replace those obligations.

Only the offline simulation and analysis dependencies are used; this release does not import cflib, ROS, Crazyswarm2 or a proprietary Vicon SDK.

## Research citations

Please cite the framework/article using [CITATION.cff](CITATION.cff), and credit
the solver and simulator used in your own experiments:

- Alberto Bemporad, *NashOpt: A Python Library for Computing Generalized Nash
  Equilibria and Solving Game-Design and Game-Theoretic Control Problems*,
  [arXiv:2512.23636](https://arxiv.org/abs/2512.23636).
- Jacopo Panerati, Hehui Zheng, SiQi Zhou, James Xu, Amanda Prorok and Angela P.
  Schoellig, *Learning to Fly—a Gym Environment with PyBullet Physics for
  Reinforcement Learning of Multi-agent Quadcopter Control*, IROS 2021,
  [doi:10.1109/IROS51168.2021.9635857](https://doi.org/10.1109/IROS51168.2021.9635857).

Research citation is requested independently of the legal license terms.
Attribution for the CC BY data and figures: Zdeněk Bouček and Miroslav Flídr,
*A Game-Theoretic MPC Framework for Cooperative and Adversarial Drone
Coordination*, companion release 1.0.0, 2026. Include the license link and
describe modifications when sharing adapted data or figures.

The included simulator wheel contains unchanged package sources and assets from the pinned commit. Its installation metadata declares the Python 3.11 dependency versions tested with this companion, replacing the upstream Python 3.12 requirements. The wheel includes the upstream MIT license and identifies its source commit.
