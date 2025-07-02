# Temporal-Continual-Learning-with-Prior-Compensation-for-Human-Motion-Prediction

This is official implementation for NeurIPS2023 paper **Temporal Continual Learning with Prior Compensation for Human Motion Prediction**.

[[paper]](https://papers.nips.cc/paper_files/paper/2023/file/cf7a83a5342befd11d3d65beba1be5b0-Paper-Conference.pdf)

We provide model checkpoints for multiple backbone models on different datasets.

### Usage

`cd {PGBIG/LTD/STSGCN/siMLPe}` and follow the instructions in `{PGBIG/LTD/STSGCN/siMLPe}/README.md`.

> ### *Visualization*
>
> For visualization, we have developed a simple [visualization-tool](https://github.com/hyqlat/PyRender-for-Human-Mesh/tree/Mesh_and_Skeleton) that can be used following the instructions in [README.md](https://github.com/hyqlat/PyRender-for-Human-Mesh/blob/Mesh_and_Skeleton/README.md#pyrenskele-for-human-skeleton).

### Citation

If you find this project useful in your research, please consider citing:

```bash
@article{tang2023temporal,
  title={Temporal continual learning with prior compensation for human motion prediction},
  author={Tang, Jianwei and Sun, Jiangxin and Lin, Xiaotong and Zheng, Wei-Shi and Hu, Jian-Fang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={65837--65849},
  year={2023}
}
```

### Acknowledgments

Our code is based on [PGBIG](https://github.com/705062791/PGBIG), [LTD](https://github.com/705062791/PGBIG), [STSGCN](https://github.com/FraLuca/STSGCN) and [siMLPe](https://github.com/dulucas/siMLPe).
