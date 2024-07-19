# siMLPe+TCL

### Requirements

---

- PyTorch >= 1.5
- Numpy
- CUDA >= 10.1
- Easydict
- pickle
- einops
- scipy
- six

### Data Preparation

---

**H3.6M**

Download the data and put them in the `./data` directory. Please redirect to [siMLPe](https://github.com/dulucas/siMLPe) for more details.

Directory structure:

```shell
data
|-- h36m
|   |-- S1
|   |-- S5
|   |-- S6
|   |-- ...
|   |-- S11
```

---

#### H3.6M

For training and testing, you can run the script:

```bash
bash run.sh
```

The code is partially based on [siMLPe](https://github.com/dulucas/siMLPe).
