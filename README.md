<h1 align="center">Do Video Models Render<br>What the Program Specifies?</h1>

<p align="center"><a href="https://alayalab.ai/"><b>Alaya Lab</b></a></p>

<p align="center">
  Zheng-Hui Huang<sup>*</sup>, Guixu Lin<sup>*</sup>, Yu-Ju Tsai, Jian-Kai Zhu, Fengbo Lan,<br>
  Yu-Lun Liu, Yung-Yu Chuang, Kaipeng Zhang<sup>†</sup>, Zhixiang Wang<sup>†</sup><br>
  <sub>* Equal contribution · † Correspondence</sub>
</p>


  <a href="https://alaya-lab.github.io/PROWBench/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
  <a href="https://alaya-lab.github.io/PROWBench/assets/PROWBench.pdf"><img src="https://img.shields.io/badge/Paper-PDF-red"></a>


<p align="center">
  <img src="assets/teaser.png" width="100%" alt="PROWBench overview">
</p>

<p align="center">
  <i>Visual similarity alone cannot tell whether the program-specified action and end state actually occur.</i>
</p>

> A benchmark for programmable world models: 170 programmatically constructed episodes and 600 proxy videos, each logged as a replayable world record of entity states and timestamped events, so generated videos can be checked against the observable consequences of program execution.

Programmable world models separate executable dynamics from visual generation, but their visual adherence to explicit rules and interactions remains insufficiently evaluated. PROWBench renders synchronized views and proxy representations (e.g., coarse 3D, bounding boxes) from engine-recorded world records across first- and third-person perspectives, and evaluates entity control, long-horizon memory, and — with two VLM-based metrics, Logic-Render Alignment and Interaction Success Rate — whether timestamped events are visually realized on the prescribed timeline.

## 📰 News

- **[2026-10-02]** [Project page](https://alaya-lab.github.io/PROWBench/) released.

## 🚀 Release Roadmap

- [x] Project page
- [x] Paper — [PDF](https://alaya-lab.github.io/PROWBench/assets/PROWBench.pdf)
- [ ] Benchmark data
- [ ] Evaluation code

## Citation

Coming soon.
