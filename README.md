# MLSCA-Bench

*A unified benchmarking framework for Machine Learning-based Side-Channel Analysis.*

MLSCA-Bench aims to standardize datasets, evaluation protocols, and reporting practices for machine learning-based side-channel analysis. It provides a common framework for developing, evaluating, and comparing ML-SCA methods under reproducible and consistent conditions.

---

## Why MLSCA-Bench?

Deep learning has become a dominant approach for profiling side-channel attacks, yet the ML-SCA ecosystem remains fragmented.

Current research often suffers from:

- Different dataset formats and preprocessing pipelines.
- Inconsistent evaluation protocols.
- Difficult-to-reproduce experimental settings.
- Incomparable performance metrics across papers.

These inconsistencies make it difficult to fairly compare methods or reproduce published results.

MLSCA-Bench addresses these issues by providing a unified benchmarking framework that enables consistent evaluation across datasets, devices, countermeasures, and machine learning models. By establishing standardized benchmarks and a common foundation, we hope to accelerate progress in machine learning-based side-channel analysis while improving the reproducibility and transparency of published research.

---

## Roadmap

The project is being developed incrementally.

Planned components include:

- [ ] Unified dataset loaders
- [ ] Common preprocessing pipeline
- [ ] Reference implementations of ML-SCA models
- [ ] Standard evaluation metrics
- [ ] Benchmark configuration system

---

## The Framework 

![Project Architecture](figures/mlscabenchnew.png)

---

## Datasets 

Below an exhaustive list of the supported datasets, more information about the datasets can be find [here.](documentation/dataset)

| Dataset | Available |
|---------|:---------:|
| [AES 2023](https://gitee.com/yuyuyu1256/sakura-g-aes-present) | ✅ |
| AES HD | ❌ |
| AES HD ext | ❌ |
| [AES HD git](https://github.com/AESHD/AES_HD_Dataset) | ✅ |
| [AES HD zaid](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA) | ✅ |
| [AES PTv2 Piñata](https://github.com/urioja/AESPTv2) | ✅ |
| [AES PTv2 Piñata MS1](https://github.com/urioja/AESPTv2) | ✅ |
| [AES PTv2 Piñata MS2](https://github.com/urioja/AESPTv2) | ✅ |
| [AES PTv2 STM32F4-D1/2/3/4](https://github.com/urioja/AESPTv2) | ✅ |
| [AES PTv2 STM32F4-D1/2/3/4 MS1](https://github.com/urioja/AESPTv2) | ✅ |
| [AES PTv2 STM32F4-D1/2/3/4 MS2](https://github.com/urioja/AESPTv2) | ✅ |
| [AES RD](https://github.com/ikizhvatov/randomdelays-traces) | ✅ |
| [ASCAD raw](https://github.com/ANSSI-FR/ASCAD) | ✅ |
| [ASCADf](https://github.com/ANSSI-FR/ASCAD) | ✅ |
| [ASCADv](https://github.com/ANSSI-FR/ASCAD) | ✅ |
| [ASCADv2](https://github.com/ANSSI-FR/ASCAD/tree/master/STM32_AES_v2) | ✅ |
| [ASCADv2r](https://github.com/ANSSI-FR/ASCAD/tree/master/STM32_AES_v2) | ✅ |
| [ASCON protected](https://zenodo.org/records/10229485) | ✅ |
| [ASCON unprotected](https://zenodo.org/records/10229484) | ✅ |
| [AT128 N & F desync](https://github.com/lxj-sjtu/TCHES2021_Pay_attention_to_the_raw_traces) | ❌ |
| [AT128 N & F sync](https://github.com/lxj-sjtu/TCHES2021_Pay_attention_to_the_raw_traces) | ❌ |
| [BLISS A](https://zenodo.org/records/5101343) | ✅ |
| [BLISS B](https://zenodo.org/records/5101343) | ✅ |
| [Chameleon BASE](https://github.com/hardware-fab/chameleon) | ✅ |
| [Chameleon CHF](https://github.com/hardware-fab/chameleon) | ✅ |
| [Chameleon DFS](https://github.com/hardware-fab/chameleon) | ✅ |
| [Chameleon MRP](https://github.com/hardware-fab/chameleon) | ✅ |
| [Chameleon RD](https://github.com/hardware-fab/chameleon) | ✅ |
| [CHES CTF 2018 aisylab](https://www.dropbox.com/scl/fi/98yxftt29sxcpu7c6c06c/ches_ctf.h5) | ✅ |
| [CHES CTF 2018 challenge 2](https://zenodo.org/records/3733418) | ✅ |
| CHES CTF 2018 original | ❌ |
| [CHES CTF 2020 HW](https://dataverse.uclouvain.be/dataset.xhtml?persistentId=doi:10.14428/DVN/W2SV5G) | ✅ |
| [CHES CTF 2020 SW](https://dataverse.uclouvain.be/dataset.xhtml?persistentId=doi:10.14428/DVN/W2SV5G) | ✅ |
| [CrossEM](https://github.com/UCdasec/CrossEM) | ✅ |
| [Curve25519](https://www.dropbox.com/scl/fi/cpoaco0zwmgzdarj2427p/ecc_datasets.zip) | ✅ |
| [curve25519cswap-arith](https://www.dropbox.com/scl/fi/cpoaco0zwmgzdarj2427p/ecc_datasets.zip) | ✅ |
| [DFS_DESYNCH](https://huggingface.co/datasets/hardware-fab/DFS_DESYNCH) | ✅ |
| [Dilithium 2025 A](https://www.scidb.cn/en/detail?dataSetId=aa3512fdbc294719af4663757fc28468) | ✅ |
| [DPAv1.1](https://dpacontest.telecom-paris.fr/tables.php) | ✅ |
| [DPAv1.2](https://dpacontest.telecom-paris.fr/tables.php) | ✅ |
| [DPAv1.3](https://dpacontest.telecom-paris.fr/tables.php) | ✅ |
| [DPAv2](https://dpacontest.telecom-paris.fr/v2/tables.php) | ✅ |
| [DPAv4.1](https://dpacontest.telecom-paris.fr/v4/rsm_traces.php) | ✅ |
| [DPAv4.2](https://dpacontest.telecom-paris.fr/v4/42_traces.php) | ✅ |
| [eShard non shuffled](https://gitlab.com/eshard/nucleo_sw_aes_masked_shuffled) | ✅ |
| [eShard shuffled](https://gitlab.com/eshard/nucleo_sw_aes_masked_shuffled) | ✅ |
| [Ge_wars](https://drive.google.com/drive/folders/1JGbphwZXQvN_tEhpBIbQ-q-pN9wkqKQ-) | ✅ |
| [Kyber](https://zenodo.org/records/15352482) | ✅ |
| Piñata SW AES | ❌ |
| [Portability](https://zenodo.org/records/19060677) | ✅ |
| [PRESENT 2021 (1–9)](https://github.com/Chair-for-Security-Engineering/DL-LA) | ✅ |
| [PRESENT 2023](https://gitee.com/yuyuyu1256/sakura-g-aes-present) | ✅ |
| [Re-encryption AESa](https://github.com/ECSIS-lab/curse_of_re-encryption) | ✅ |
| [Re-encryption AESb](https://github.com/ECSIS-lab/curse_of_re-encryption) | ✅ |
| [Re-encryption AESc](https://github.com/ECSIS-lab/curse_of_re-encryption) | ✅ |
| [Re-encryption AESd](https://github.com/ECSIS-lab/curse_of_re-encryption) | ✅ |
| [Re-encryption NRTP](https://github.com/ECSIS-lab/curse_of_re-encryption) | ✅ |
| [Re-encryption SHAKE](https://github.com/ECSIS-lab/curse_of_re-encryption) | ✅ |
| [REASSURE Curve25519](https://zenodo.org/records/3609789) | ✅ |
| [SCAAML ECC M0](https://github.com/google/scaaml/tree/main/papers/2024/GPAM) | ✅ |
| [SCAAML ECC M1](https://github.com/google/scaaml/tree/main/papers/2024/GPAM) | ✅ |
| [SCAAML ECC M2](https://github.com/google/scaaml/tree/main/papers/2024/GPAM) | ✅ |
| [SCAAML ECC M3](https://github.com/google/scaaml/tree/main/papers/2024/GPAM) | ✅ |
| [SMAesh Artix-7](https://repository.tugraz.at/records/bk4fx-rbh46) | ✅ |
| [SMAesh Spartan-6](https://repository.tugraz.at/records/bk4fx-rbh46) | ✅ |
| [TeSCASE ECC EM](https://chest.coe.neu.edu/?current_page=POWER_TRACE_LINK&software=emeccarm) | ✅ |
| [TeSCASE GPU AES Power](https://chest.coe.neu.edu/?current_page=POWER_TRACE_LINK&software=gpupower) | ✅ |
| [TeSCASE GPU AES Timing](https://chest.coe.neu.edu/?current_page=POWER_TRACE_LINK&software=gputiming) | ✅ |
| [TeSCASE Keccak](https://chest.coe.neu.edu/?current_page=POWER_TRACE_LINK&software=ptkeccak) | ✅ |
| [TeSCASE MAS AES](https://chest.coe.neu.edu/?current_page=POWER_TRACE_LINK&software=ptmasked) | ✅ |
| [TeSCASE UN AES](https://chest.coe.neu.edu/?current_page=POWER_TRACE_LINK&software=ptunmasked) | ✅ |
| [wolfSSL Ed25519](https://github.com/leoweissbart/MachineLearningBasedSideChannelAttackonEdDSA) | ✅ |
| [X-DeepSCA](https://github.com/SparcLab/X-DeepSCA) | ✅ |
| Xoodyak | ❌ |



## Models 

Coming soon.


## Metrics

Coming soon. 

---

## Contributing

MLSCA-Bench is an open research initiative developed by the [ESSEC group](https://www.unibw.de/essec) at the [University of the Bundeswehr Munich](https://www.unibw.de/home-en). We welcome feedback, collaborations, and contributions from the side-channel analysis and machine learning communities.

If you would like to contribute, please open an issue, submit a pull request or contact us at `iris.jimenez@unibw.de` or `michael.hutter@unibw.de`. 

---

## License

See the LICENSE file for details.

---

## Citation 

if you use MLSCA-Bench, please cite us: 

```
@unpublished{MLASCA_Bench_2026,
  author       = {Iris Dania Jimenez and Michael Hutter},
  title        = {MLSCA-Bench: A Unified Benchmarking Framework for Machine Learning-based Side-Channel Analysis},
  year         = {2026},
  note         = {Manuscript submitted for publication},
}
```