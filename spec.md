# EEG Emotion Recognition Library

## Project Specification and Research Plan

**Version:** 0.1  
**Status:** Proposed / Working Specification  
**Date:** November 2026  
**Project name:** Undecided  
**Project type:** Open-source scientific Python library

---

# 1. Project Overview

## 1.1 Vision

The project aims to develop a modular, extensible Python library capable of recognizing human emotional states from electroencephalography (EEG) signals through signal processing, feature extraction, and machine learning.

The library will support both broad emotional dimensions and specific emotion categories, while providing rigorous tools for training, testing, and comparing recognition models against established scientific research.

## 1.2 Primary Objective

**Input:** Multichannel EEG signal.

**Output:** An estimate of the subject's emotional state.

The system will support two primary forms of recognition:

**Dimensional emotion recognition**

Prediction of continuous emotional dimensions, particularly:

- Valence — unpleasant to pleasant.
- Arousal — low to high activation.
- Dominance — an optional additional dimension.

**Specific emotion recognition**

Classification of identifiable emotional categories, potentially including:

- Happiness / joy.
- Sadness.
- Fear.
- Anger.
- Disgust.
- Surprise.
- Neutrality.
- Other categories supported by suitable datasets.

Not every trained model will necessarily recognize all categories. The available outputs will depend on its training labels and the task for which it was developed.

The library will estimate emotional states statistically. EEG does not directly measure emotion, and predicted categories must not be treated as certain descriptions of a person's internal experience.

## 1.3 Secondary Objectives

The library will also provide:

1. Reusable EEG preprocessing and feature-extraction components.
2. Classical and neural-network model implementations.
3. Advanced decomposition and representation-learning methods.
4. A unified interface for model training and prediction.
5. Reproducible testing and benchmark execution.
6. Systematic comparison with results reported in scientific literature.
7. Tools for studying model generalization, robustness, and reliability.

**Scientific evaluation is a fundamental quality requirement of the library, but the principal deliverable remains a functioning EEG emotion-recognition system.**

## 1.4 Core Research Questions

**RQ1 — Recognition capability**

To what extent can EEG signals support the identification of broad and specific emotional states?

**RQ2 — Model architecture**

Which machine-learning and deep-learning approaches provide the strongest performance under consistent experimental conditions?

**RQ3 — EEG representation**

Can advanced preprocessing, spectral analysis, and tensor decomposition improve emotion recognition compared with conventional features?

**RQ4 — Generalization**

How well do models trained on particular participants, stimuli, and datasets generalize to unseen examples?

**RQ5 — Scientific comparison**

How does the developed system compare with established research implementations when evaluated using equivalent datasets, labels, and testing procedures?

These questions will guide the design of the library and its experimental development.

---

# 2. Intended Users and Scope

## 2.1 Target Users

The library is intended for:

**Machine-learning researchers**

Researchers developing EEG-based classification and regression algorithms.

**Neuroscience and affective-computing researchers**

Researchers investigating relationships between brain activity and emotional experiences.

**Students**

Students exploring EEG preprocessing, neural networks, emotion recognition, and advanced mathematical representations.

**Software developers**

Developers who need to obtain emotional-state predictions from EEG recordings using a Python API.

## 2.2 Core User Scenarios

**UC-01 — Emotion prediction**

A developer supplies an EEG recording to a previously trained model and obtains predicted emotional categories or dimensional values.

**UC-02 — Model training**

A researcher uses a labeled EEG dataset to train an emotion-recognition model.

**UC-03 — Specific emotion classification**

A researcher trains a classifier to distinguish fear, sadness, happiness, and neutrality from EEG.

**UC-04 — Preprocessing comparison**

A researcher compares conventional band-power features against wavelet and tensor-based representations.

**UC-05 — Model benchmarking**

A researcher evaluates EEGNet, SVM, and another architecture under identical subject-independent test conditions.

**UC-06 — Scientific reproduction**

A researcher reproduces an experimental configuration from the literature and investigates differences between the reproduced and published results.

## 2.3 Initial Scope

The library will include:

- EEG recording import and validation.
- Signal preprocessing.
- Feature extraction.
- Tensor and time–frequency representations.
- Model training and inference.
- Discrete emotion classification.
- Dimensional emotion regression.
- Model persistence.
- Benchmark execution.
- Performance reporting.
- Experiment configuration management.

## 2.4 Excluded from Version 0.1

The initial release will not include:

- EEG hardware drivers.
- Direct EEG device acquisition.
- Live streaming or real-time processing infrastructure.
- Web or desktop interfaces.
- VR-specific components.
- Medical diagnosis.
- Claims of universal emotion recognition across all populations and environments.

These features may be developed independently in later releases.

---

# 3. Emotion Recognition Tasks and Labeling Strategy

A major architectural requirement is that the library must support different definitions of emotional states.

Datasets frequently use different categories, rating scales, and experimental procedures. These differences cannot be ignored when developing general-purpose recognition software.

## 3.1 Dimensional Classification

The first recognition family will use valence and arousal.

Example tasks:

- High versus low valence.
- High versus low arousal.
- Continuous valence regression.
- Continuous arousal regression.

DEAP will serve as the primary initial dataset for these experiments.

## 3.2 Specific Emotion Classification

The second recognition family will identify individual emotional categories.

Initially, the library should support multiple classification configurations:

| Task        | Example categories                                                              | Primary dataset |
| ----------- | ------------------------------------------------------------------------------- | --------------- |
| Three-class | Positive, neutral, negative                                                     | SEED            |
| Four-class  | Happiness, sadness, fear, neutral                                               | SEED-IV         |
| Seven-class | Joy, inspiration, tenderness, fear, disgust, sadness, neutral                   | EmoEEG-MC       |
| Nine-class  | Amusement, inspiration, joy, tenderness, anger, fear, disgust, sadness, neutral | FACED           |

These tasks are related but not interchangeable.

For example, a four-class model trained on SEED-IV cannot automatically be interpreted as recognizing anger or disgust.

**Specific emotion recognition is a core development objective, not merely an optional future feature.**

The initial version should establish at least one specific-emotion benchmark, with SEED-IV as the preferred starting point.

## 3.3 Hierarchical Emotion Classification

A potential extension is to organize emotional categories into a hierarchy.

For example:

```
Emotional state
├── Positive
│   ├── Joy
│   ├── Amusement
│   └── Tenderness
│
├── Negative
│   ├── Fear
│   ├── Sadness
│   ├── Anger
│   └── Disgust
│
└── Neutral
```

The model could predict both a broad category and a specific emotional state.

This may allow different levels of recognition depending on the confidence and granularity supported by the data.

However, such a hierarchy is a modeling assumption. Emotion categories can overlap, and the selected relationships must be documented rather than treated as a universal psychological taxonomy.

## 3.4 Multi-Task Learning

A possible advanced architecture will use a shared EEG encoder with several prediction heads:

```
EEG Input
    |
Shared Encoder
    |
    +-- Valence Prediction
    |
    +-- Arousal Prediction
    |
    +-- Broad Emotion Classification
    |
    +-- Specific Emotion Classification
```

The shared encoder may learn useful representations across related prediction tasks.

Separate losses will be used for classification and regression, and losses will only be calculated where appropriate ground-truth labels exist.

This design will be considered after reliable single-task baselines have been established.

## 3.5 Emotion Intensity and Multi-Label Recognition

Some datasets provide ratings for several emotions simultaneously.

For example, a participant may report both fear and sadness rather than experiencing exactly one emotion.

The library should therefore permit future models to predict:

- The intensity of individual emotions.
- Multiple emotion scores simultaneously.
- Continuous ratings instead of only mutually exclusive classes.

This represents a multi-label or multi-output regression problem rather than conventional single-label classification.

FACED and EmoEEG-MC provide self-reported emotional intensity information that makes this direction worth investigating.

## 3.6 Label Quality and Ground Truth

The project must distinguish:

**Stimulus-assigned labels:** Labels representing emotions an experimental stimulus was designed to elicit.

**Self-reported labels:** Labels based on the participant's reported emotional experience.

These may disagree.

The library must preserve the original labels and permit alternative label mappings, thresholds, and supervision strategies.

Trial-level labels must not automatically be interpreted as independently measured labels for every short EEG window.

---

# 4. Dataset Strategy

## 4.1 Selected Datasets

| Dataset   | Participants | EEG channels | Main use                                             |
| --------- | -----------: | -----------: | ---------------------------------------------------- |
| DEAP      |           32 |           32 | Valence/arousal recognition                          |
| SEED      |           15 |           62 | Three-class recognition                              |
| SEED-IV   |           15 |           62 | Four specific emotions                               |
| FACED     |          123 |           32 | Nine-class recognition and generalization            |
| EmoEEG-MC |           60 |           64 | Seven-class recognition and cross-context evaluation |
| DREAMER   |           23 |           14 | Independent dimensional-emotion benchmark            |

These datasets provide complementary tasks. They will not be treated as a single homogeneous training dataset.

## 4.2 Initial Dataset Sequence

**Phase A — DEAP**

Establish the first complete EEG processing and model-training pipeline.

Train and evaluate binary valence and arousal classifiers.

**Phase B — SEED-IV**

Introduce specific emotion recognition using happiness, sadness, fear, and neutral classes.

Evaluate whether the existing processing architecture supports a different electrode arrangement and multiclass classification.

**Phase C — FACED**

Investigate more detailed emotion categories and larger-scale cross-subject performance.

Compare stimulus-assigned classifications with self-report-based analyses where supported.

**Phase D — EmoEEG-MC**

Investigate whether models generalize across different emotional elicitation contexts, such as watching videos and mentally imagining emotional situations.

This provides an additional form of generalization beyond subject-independent evaluation.

## 4.3 Data Integrity Requirements

Every sample must retain:

- Dataset identifier and version.
- Participant identifier.
- Session identifier where available.
- Trial identifier.
- Stimulus identifier.
- EEG channel names and ordering.
- Sampling rate and signal units.
- Original emotion labels.
- Derived task-specific labels.
- Preprocessing history.

These identifiers are essential to preventing evaluation leakage and reconstructing experiments.

Dataset licenses and access restrictions must be respected. The library will not bundle restricted research recordings.

---

# 5. Signal Processing and Machine-Learning Design

## 5.1 General Pipeline

```
EEG Input
    |
Signal Validation
    |
Preprocessing
    |
Feature Extraction / Representation Learning
    |
Emotion Recognition Model
    |
Emotional State Prediction
```

Training and inference will use consistent preprocessing configurations.

## 5.2 Preprocessing

Initial methods will include:

- Channel selection and referencing.
- Band-pass and notch filtering.
- Resampling.
- Window segmentation.
- Normalization.
- Basic signal-quality evaluation.
- Optional artifact correction.

Window length and preprocessing settings will be configurable.

The system must distinguish transformations with fixed parameters from transformations that learn parameters using training recordings.

## 5.3 Conventional Features

The initial feature-extraction module will support:

- Power Spectral Density.
- Frequency-band power.
- Differential entropy.

These are essential baselines because they are widely used in published EEG emotion-recognition experiments.

## 5.4 Advanced Representations

The research-oriented feature module will investigate:

**Wavelet decomposition**

Multiresolution analysis of time-varying EEG frequency components.

**Tucker decomposition**

Low-dimensional representations of channel–frequency–time EEG tensors.

**CP/PARAFAC**

Factorization into interpretable spatial, spectral, and temporal components.

**Riemannian covariance analysis**

Representations based on relationships between EEG channels.

**Variational Mode Decomposition**

Data-adaptive separation of oscillatory components.

The first advanced method implemented will be Tucker decomposition, supported by wavelet-derived time–frequency representations.

Additional methods will be prioritized according to experimental results and development resources.

## 5.5 Model Architectures

| Model                     | Role                                     |
| ------------------------- | ---------------------------------------- |
| Logistic Regression       | Simple linear baseline                   |
| SVM                       | Classical EEG classification baseline    |
| EEGNet                    | Compact neural-network baseline          |
| TSception                 | EEG emotion-specific CNN                 |
| DGCNN                     | Graph-based model                        |
| EEG Conformer             | Advanced attention-based model           |
| Tensor-feature classifier | Project-specific representation research |

SVM and EEGNet will be implemented first.

TSception and DGCNN will provide more advanced comparisons, especially for specific emotion classification.

## 5.6 Confidence and Uncertainty

A further requirement is that model outputs should make uncertainty accessible.

Relevant methods include:

- Probability calibration.
- Entropy of class predictions.
- Confidence-threshold-based abstention.
- Detection of low-quality or unsupported inputs.

A model may be uncertain whether an EEG recording represents fear or sadness. In appropriate circumstances, reporting uncertainty may be more useful than forcing a definitive category.

These methods will initially be considered research features. Model probabilities must not be represented as guaranteed psychological certainty.

---

# 6. Architecture and Technology

## 6.1 Architectural Principles

The library will be modular, dataset-independent, extensible, and reproducible.

Individual components must be replaceable without requiring extensive modifications to unrelated modules.

## 6.2 Proposed Modules

```
src/<package_name>/
    data/
        loaders/
        adapters/
        metadata/

    preprocessing/
        filters/
        artifacts/
        segmentation/

    features/
        spectral/
        wavelets/
        tensors/
        covariance/

    models/
        classical/
        neural/
        multitask/

    training/
        trainers/
        optimization/

    evaluation/
        splits/
        metrics/
        statistical_tests/
        literature_benchmarks/
        robustness/

    experiments/
        configurations/
        tracking/
        reporting/

    pipeline/
        inference/
        serialization/

tests/
examples/
docs/
```

The package name is a placeholder.

## 6.3 Primary Interfaces

**EEGRecording:** Standardized EEG signal and metadata.

**DatasetAdapter:** Access to public EEG dataset formats.

**EEGPreprocessor:** Signal preprocessing operations.

**FeatureExtractor:** Transformation of EEG into model-ready features.

**EmotionEstimator:** Common interface for training and prediction.

**EmotionPipeline:** Composition of preprocessing, extraction, and prediction.

**BenchmarkRunner:** Reproducible execution of evaluation protocols.

**ExperimentRegistry:** Storage of experiment definitions, model configurations, and results.

## 6.4 Technology Stack

**Language:** Python 3.11+

**Scientific computing:** NumPy, SciPy, MNE-Python.

**Machine learning:** scikit-learn, PyTorch.

**Advanced processing:** PyWavelets, TensorLy, pyRiemann.

**EEG model implementations:** Braindecode and selected TorchEEG components.

**Testing:** pytest.

**Code quality:** Ruff and optional mypy.

**Packaging:** pyproject.toml with setuptools.

**Documentation:** MkDocs or Sphinx.

**Continuous integration:** GitHub Actions.

Advanced dependencies should be optional wherever possible.

---

# 7. Scientific Benchmarking and Literature Comparison

## 7.1 Importance of Benchmarking

Benchmarking will be integrated into development from the beginning.

It serves three purposes:

1. **Verify the implementation:** Determine whether established models behave as expected.
2. **Evaluate model quality:** Quantify recognition performance on independent test data.
3. **Establish scientific context:** Compare the project's findings with published results under clearly documented conditions.

Although reproduction is not the project's main objective, the library must not claim improved performance without appropriate reference experiments.

Scientific comparison is therefore a mandatory validation activity for all major model-development stages.

## 7.2 Literature Benchmark Registry

The project will maintain a concise record of relevant published benchmarks, including the scientific source, dataset, recognition task, model, evaluation methodology, and reported performance.

Published results will be compared with the library's own experiments to assess model performance and reproducibility. Comparisons will clearly distinguish results obtained under equivalent experimental conditions from those using different protocols.

---

## 7.3 Literature Reference A — LibEER

**Publication:** Liu et al., _LibEER: A Comprehensive Benchmark and Algorithm Library for EEG-Based Emotion Recognition_, 2025.

Repository: https://github.com/XJTU-EEG/LibEER

The following accuracy values come from LibEER's reported cross-subject benchmark.

| Model     | SEED (3 classes) | SEED-IV (4 classes) | DEAP Valence | DEAP Arousal |
| --------- | ---------------: | ------------------: | -----------: | -----------: |
| SVM       |           37.07% |              28.98% |       49.58% |       51.48% |
| EEGNet    |           38.19% |              28.19% |       52.36% |       48.94% |
| TSception |           45.60% |              34.19% |       54.44% |       45.90% |
| DGCNN     |           60.87% |              42.54% |       49.91% |       49.91% |

These figures are reference values from the original LibEER protocol, not universal performance guarantees.

### Why This Matters

These results establish initial reference points for both generic and specific emotion recognition.

SEED-IV is particularly important because its four classes correspond to distinct emotional categories rather than broad positive or negative affect.

The results also suggest that EEGNet is not necessarily the strongest model for fine-grained emotion classification under the reported settings.

### Proposed Project Comparison

The project should reproduce at least one classical and one neural-network baseline from LibEER.

For specific emotion recognition, the initial comparison should include:

- SVM on SEED-IV.
- EEGNet on SEED-IV.
- TSception or DGCNN on SEED-IV.
- The project's proposed representation-based classifier.

The same training and testing partitions must be used when comparing internal models.

---

## 7.4 Literature Reference B — EEGEmoLib

**Publication:** _EEG-Based Emotion Recognition for Machine Intelligence: A Survey with an Open-Source Python Toolbox_, 2026.

DOI: https://doi.org/10.1093/nsr/nwag460

Selected subject-independent results from its published Table 9:

| Task         | Method                             | Accuracy |
| ------------ | ---------------------------------- | -------: |
| SEED         | ATDD-LSTM + differential entropy   |    90.5% |
| DEAP arousal | ATDD-LSTM + differential entropy   |    72.3% |
| DEAP valence | ACRNN + PSD                        |    75.3% |
| SEED-V       | DeepConvNet + differential entropy |    72.5% |

These results differ substantially from some LibEER results.

Such differences motivate the requirement to document exact experimental settings before making direct comparisons.

The project's benchmark system must distinguish between comparing literature results descriptively and performing a controlled comparison using identical data and evaluation protocols.

---

## 7.5 Literature Reference C — Fine-Grained Emotion Recognition

### FACED

**Publication:** Chen et al., _A Large Finer-grained Affective Computing EEG Dataset_, Scientific Data, 2023.

DOI: https://doi.org/10.1038/s41597-023-02650-w

FACED contains 123 participants and nine emotional categories, including four positive and four negative specific emotions.

The authors reported 78.8% intra-subject and 69.3% cross-subject accuracy for a **binary positive-versus-negative classification task**, not for nine-class classification.

These figures must not be used as nine-class benchmark targets.

### FACED Methodological Reanalysis

**Publication:** Gerster et al., _Stimulus Identity Rather Than Emotion Drives EEG Classification on the FACED Dataset_, bioRxiv preprint, 2026.

DOI: https://doi.org/10.64898/2026.06.12.731889

The authors' released nine-class cross-subject LinearSVC analyses report approximately:

| Evaluation                            | Reported accuracy |
| ------------------------------------- | ----------------: |
| Stimulus-assigned labels              |             39.4% |
| Participant-derived subjective labels |             26.8% |
| Single video per emotion              |             44.9% |

These numbers refer to the reanalysis's own procedures and should not be interpreted as directly comparable with the original FACED binary classification results.

The findings raise a serious question: is the model recognizing emotional experience or identifying information related to the stimulus?

### Implications

Fine-grained classification must be assessed against both assigned emotion labels and participants' own reports when available.

A high nine-class score is not sufficient evidence that nine emotional experiences are being recognized.

---

## 7.6 Literature Reference D — EmoEEG-MC

**Publication:** Xu et al., _A Multi-Context Emotional EEG Dataset for Cross-Context Emotion Decoding_, Scientific Data, 2025.

DOI: https://doi.org/10.1038/s41597-025-05349-2

The dataset includes 60 participants and seven emotional categories elicited through both video viewing and mental imagery.

The authors' SVM validation reported accuracy of 66.7% for binary positive/negative recognition and 28.9% for seven-class recognition in their reported experimental comparison.

These provide additional literature reference points, subject to matching the original task and protocol.

### Why This Dataset Is Interesting

It enables a new research question:

**Does a model recognize an emotional state independently of the method used to evoke that emotion?**

A model trained on videos could be evaluated on imagery recordings.

This is called cross-context generalization and represents a stronger test of emotional representation than ordinary within-context classification.

Cross-context experiments will be considered an advanced evaluation track.

---

## 7.7 Literature Reference E — Subject Identity Confounding

**Publication:** Pandilova et al., _Subject Identity Confounds qEEG Emotion Recognition on DEAP and DREAMER_, Sensors, 2026.

DOI: https://doi.org/10.3390/s26175327

Selected DEAP results:

| Evaluation procedure               | Valence ROC-AUC | Arousal ROC-AUC |
| ---------------------------------- | --------------: | --------------: |
| Randomly pooled epochs             |           0.689 |           0.711 |
| Trial-grouped, participant overlap |           0.625 |           0.653 |
| Participant-independent            |           0.493 |           0.447 |

These results relate to the specific quantitative EEG feature/model setting studied.

They demonstrate that participant overlap can substantially affect apparent recognition performance.

The confidence intervals for the participant-independent results included chance performance, so the values should not be interpreted as evidence of reliably below-chance prediction.

The project must therefore treat leakage-aware subject-independent evaluation as a mandatory scientific requirement.

---

# 8. Evaluation Protocol

## 8.1 Primary Evaluation

**Subject-independent evaluation will be the primary benchmark for generalization.**

The framework will support:

- Grouped K-fold cross-validation.
- Leave-one-subject-out evaluation.
- Nested cross-validation for hyperparameter selection.

EEG windows belonging to the same trial must not be separated across training and test groups.

For trial-labeled datasets, trial-level results will be the primary reporting unit. Window-level metrics may also be reported for diagnostics.

## 8.2 Secondary Evaluation Protocols

**Subject-dependent, trial-independent evaluation**

Measures performance on new trials from familiar participants.

**Cross-session evaluation**

Measures consistency across recording sessions.

**Stimulus-independent evaluation**

Measures transfer to new emotional stimuli.

**Cross-context evaluation**

Measures transfer between emotion-elicitation procedures, such as videos and imagery.

**Cross-dataset evaluation**

Measures transfer across independently collected EEG datasets.

Each protocol must be separately identified in benchmark reports.

## 8.3 Metrics for General Emotion Recognition

For valence/arousal classification:

- Balanced accuracy.
- Macro-F1.
- ROC-AUC.
- Ordinary accuracy for published comparisons.

For regression:

- MAE.
- RMSE.
- Concordance correlation coefficient where appropriate.

## 8.4 Metrics for Specific Emotion Recognition

Fine-grained classification requires additional analysis.

Mandatory metrics:

- Balanced accuracy.
- Macro-F1.
- Per-class precision and recall.
- Confusion matrix.
- Participant-level performance distribution.

The benchmark must show which emotions are confused with each other.

For example, if fear and sadness are frequently confused, a single overall accuracy score does not adequately describe the system.

The confusion matrix is therefore a mandatory output for every discrete-emotion benchmark.

## 8.5 Additional Evaluation Concepts

**Hierarchical accuracy**

For hierarchical models, measure broad-category performance separately from specific-category performance.

**Emotion-intensity regression**

Where continuous self-report labels exist, evaluate the error in predicting emotion intensities.

**Calibration**

Assess whether predicted probabilities correspond to observed classification frequencies.

**Abstention**

Evaluate whether rejecting uncertain predictions improves reliability on the remaining examples, while reporting how many predictions were rejected.

**Out-of-distribution behavior**

Investigate responses to unfamiliar participants, stimuli and recording conditions.

These evaluations should be introduced progressively, with the core subject-independent benchmarks taking priority.

---

# 9. Controlled Model Comparisons

## 9.1 Baseline Comparisons

Every major experimental study should include appropriate reference methods.

Initial benchmark:

| Representation                              | Model            | Purpose                              |
| ------------------------------------------- | ---------------- | ------------------------------------ |
| Majority-class or training-prior prediction | Trivial baseline | Establish basic reference            |
| PSD                                         | SVM              | Classical baseline                   |
| Differential entropy                        | SVM              | Established EEG features             |
| Raw EEG                                     | EEGNet           | Neural-network baseline              |
| Raw EEG                                     | TSception        | Emotion-specific neural architecture |
| Time–frequency tensor                       | Tucker + SVM     | Project-specific representation      |

A new model should not be declared superior because its score exceeds a result from a paper using different preprocessing or data splits.

## 9.2 Comparing Tensor Decomposition

The primary tensor experiment will compare:

1. Standard spectral features.
2. Flattened time–frequency features.
3. PCA-compressed features.
4. Tucker-compressed features.
5. Potentially CP/PARAFAC-compressed features.

Where possible, comparisons will use the same classifier and similar feature dimensionality.

This will help isolate whether performance changes are caused by the representation or the model architecture.

## 9.3 Required Ablation Experiments

Selected experiments will investigate:

- Different EEG frequency bands.
- Different window durations.
- Different electrode subsets.
- Different tensor ranks.
- Wavelet decomposition versus ordinary spectral features.
- Model performance with and without selected preprocessing operations.

Ablations must use the same outer evaluation partitions.

## 9.4 Statistical Analysis

Reported improvements should include:

- Differences between methods on the same participants.
- Confidence intervals based on participant-aware resampling.
- Repeated-seed variability.
- Paired statistical comparisons where appropriate.
- Multiple-testing correction for families of comparisons.

A small numerical improvement without statistical support should be reported as preliminary or inconclusive.

## 9.5 Sanity Checks

The benchmark framework will include:

- Trial-level label-permutation controls.
- Participant-overlap validation.
- Majority-class and prior-only predictors.
- Checks for inconsistent label mappings.
- Analysis of stimulus-specific information where possible.
- Confirmation that decompositions and normalization have not been fitted on forbidden test data.

---

# 10. Benchmark Success Criteria

The library will have separate engineering and scientific acceptance criteria.

## 10.1 Engineering Success

The library must reliably:

- Load EEG recordings.
- Preprocess EEG signals.
- Extract features.
- Train models.
- Predict emotional states.
- Save trained models.
- Execute benchmark protocols.
- Produce reproducible evaluation reports.

## 10.2 Scientific Validation

The following evidence should be available:

**Baseline reproduction**

At least one classical and one neural-network reference experiment from existing literature should be reproduced, or any unsuccessful reproduction should be documented with the identified differences.

For a closely matched LibEER experiment, a difference of approximately five accuracy percentage points may be used as an initial diagnostic tolerance. It is not a universal statistical acceptance margin.

**Dimensional emotion recognition**

At least one valence/arousal model should be evaluated on held-out participants.

**Specific emotion recognition**

At least one model should be evaluated on a dataset containing distinct emotions, preferably SEED-IV initially.

**Controlled model comparison**

Proposed methods must be compared against appropriate baselines under identical experimental partitions.

**Generalization**

At least one experiment must measure performance on previously unseen participants.

**Uncertainty**

Main results must be accompanied by participant-level variability or confidence intervals.

There is no requirement to achieve 90% classification accuracy or outperform every published model.

An inconclusive or negative result remains scientifically useful if the experiment is valid and reproducible.

---

# 11. Software Testing and Quality Assurance

## 11.1 Unit Testing

Tests will cover:

- Signal and metadata validation.
- File loading.
- Digital filtering.
- Segmentation and window shapes.
- Frequency-domain feature computation.
- Tensor shapes and reconstruction behavior.
- Model input/output compatibility.
- Serialization.
- Invalid-input handling.

## 11.2 Integration Testing

Complete pipelines must be tested:

**Load → Preprocess → Extract → Train → Predict → Evaluate**

Tests should include synthetic signals and small reproducible fixtures.

## 11.3 Benchmark Testing

The evaluation framework must verify:

- Train/test subject disjointness.
- Trial-level grouping.
- Correct hyperparameter selection boundaries.
- Training-only fitted transformations.
- Reproducible result generation.
- Correct metric calculations.

## 11.4 Performance Testing

The library should record:

- Preprocessing runtime.
- Feature-extraction runtime.
- Model training time.
- Prediction latency.
- Peak memory consumption.
- Number of model parameters or representation dimensions.

These are secondary engineering measures, but important when comparing complex decomposition methods against simpler approaches.

---

# 12. Research Roadmap and Milestones

The project will be developed through six milestones. **Testing and literature comparison will accompany model development throughout the project rather than being postponed until the end.**

## Milestone 1 — Research Foundation and Dataset Infrastructure

**Objective:** Establish the architecture and data foundations.

Tasks:

- Review datasets and reference literature.
- Define dimensional and specific-emotion targets.
- Develop EEG data structures.
- Implement DEAP loading.
- Establish experiment metadata.
- Define subject-independent splits.
- Prepare the initial literature benchmark registry.

**Deliverables:** Dataset adapter, project structure, dataset validation tests, initial benchmark definitions.

**Completion criterion:** DEAP recordings and labels can be loaded consistently, and reproducible participant-disjoint splits can be generated.

## Milestone 2 — Preprocessing and First Models

**Objective:** Create the first complete recognition pipeline.

Tasks:

- Implement filtering and segmentation.
- Implement PSD and differential entropy.
- Integrate SVM and logistic regression.
- Train the first DEAP valence/arousal classifiers.
- Calculate initial benchmark metrics.
- Compare results with selected literature baselines.

**Deliverables:** Functional classical ML recognition pipeline and first evaluation report.

**Completion criterion:** At least one model can predict an emotional target from held-out EEG data under a documented protocol.

## Milestone 3 — Neural Networks and Specific Emotion Recognition

**Objective:** Expand recognition capabilities beyond basic dimensional classification.

Tasks:

- Integrate EEGNet.
- Implement neural-network training and checkpoints.
- Integrate SEED-IV.
- Train four-class emotion classifiers.
- Compare classical models with neural networks.
- Reproduce or investigate selected LibEER reference results.
- Report class-specific errors and confusion matrices.

**Deliverables:** Neural-network training module, specific-emotion recognition, literature comparison report.

**Completion criterion:** The library can train and evaluate at least one neural model for a specific-emotion classification task.

## Milestone 4 — Advanced Feature Extraction and Tensor Analysis

**Objective:** Investigate the main methodological research ideas.

Tasks:

- Implement wavelet-based features.
- Construct time–frequency EEG tensors.
- Implement Tucker decomposition.
- Add PCA comparison.
- Run representation and rank ablations.
- Compare tensor-based features against established baselines.

**Deliverables:** Advanced representation module and controlled experimental comparisons.

**Completion criterion:** The project provides reproducible evidence of whether the initial tensor approach improves, matches or reduces performance relative to appropriate controls.

## Milestone 5 — Comprehensive Benchmarking and Generalization

**Objective:** Establish the reliability and broader relevance of the recognition methods.

Tasks:

- Integrate FACED or EmoEEG-MC.
- Compare performance across additional emotion categories.
- Execute subject-independent evaluations.
- Investigate stimulus and/or context generalization.
- Perform selected statistical tests.
- Evaluate uncertainty and robustness.
- Compare results with additional published references.

**Deliverables:** Comprehensive benchmark report, fine-grained recognition analysis, generalization results.

**Completion criterion:** At least two relevant datasets have been evaluated with documented protocols, including a specific-emotion task and participant-independent testing.

## Milestone 6 — Library Stabilization and Research Release

**Objective:** Prepare a documented and reusable first release.

Tasks:

- Stabilize public interfaces.
- Complete unit and integration tests.
- Finalize model serialization.
- Prepare installation documentation and usage examples.
- Publish reproducible benchmark configurations.
- Complete the scientific comparison report.
- Document limitations and research conclusions.

**Deliverables:** Library Version 0.1, API documentation, benchmark configurations, technical research report.

**Completion criterion:** Another developer can install the library, train an example model, generate predictions and reproduce the documented evaluation procedure using independently obtained datasets.

---

# 13. Scientific Risks and Mitigation

| Risk                              | Effect                                 | Mitigation                                               |
| --------------------------------- | -------------------------------------- | -------------------------------------------------------- |
| Small participant counts          | Poor generalization                    | Subject-independent evaluation and uncertainty estimates |
| Weak emotion labels               | Ambiguous model targets                | Preserve self-reports and assigned labels separately     |
| Participant leakage               | Inflated accuracy                      | Grouped evaluation and automated leakage checks          |
| Stimulus recognition              | Misleading emotion classification      | Stimulus-held-out and label-concordance studies          |
| Emotion-class imbalance           | Hidden failures on individual emotions | Balanced accuracy and per-class reporting                |
| Overly complex models             | Overfitting                            | Classical baselines and repeated runs                    |
| Tensor instability                | Inconsistent features                  | Rank selection and component-stability analysis          |
| Dataset mismatch                  | Invalid comparisons                    | Explicit preprocessing and label documentation           |
| Misleading literature comparisons | Unsupported superiority claims         | Protocol-specific benchmark registry                     |

---

# 14. Definition of Done — Version 0.1

The first release will be considered complete when:

1. The library supports standardized EEG recording input.
2. A reusable preprocessing pipeline is available.
3. Conventional spectral feature extraction is implemented.
4. At least one tensor-based feature representation is implemented.
5. SVM and EEGNet can be trained and used for inference.
6. Valence/arousal recognition is supported.
7. Specific emotion classification is supported on at least one appropriate dataset.
8. Models can be saved and loaded.
9. Subject-independent evaluation is implemented.
10. Benchmark reports include comparisons against literature reference points.
11. Core experiments can be reproduced from stored configurations.
12. Automated tests and developer documentation are available.

Scientific performance will be reported separately from the engineering completion criteria.

The project must not claim reliable universal emotion recognition merely because the library is technically functional.

---

# 15. Scientific Literature

The following references form the initial research foundation.

## Emotion Datasets

**Koelstra et al. (2012).**  
_DEAP: A Database for Emotion Analysis Using Physiological Signals._  
IEEE Transactions on Affective Computing.  
https://doi.org/10.1109/T-AFFC.2011.15

**Chen et al. (2023).**  
_A Large Finer-grained Affective Computing EEG Dataset._  
Scientific Data.  
https://doi.org/10.1038/s41597-023-02650-w

**Xu et al. (2025).**  
_A Multi-Context Emotional EEG Dataset for Cross-Context Emotion Decoding._  
Scientific Data.  
https://doi.org/10.1038/s41597-025-05349-2

## Neural-Network Architectures

**Lawhern et al. (2018).**  
_EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces._  
Journal of Neural Engineering.  
https://doi.org/10.1088/1741-2552/aace8c

**Ding et al. (2022).**  
_TSception: Capturing Temporal Dynamics and Spatial Asymmetry From EEG for Emotion Recognition._  
IEEE Transactions on Affective Computing.  
https://doi.org/10.1109/TAFFC.2022.3169001

## Signal and Tensor Decomposition

**Cong et al. (2015).**  
_Tensor Decomposition of EEG Signals: A Brief Review._  
Journal of Neuroscience Methods.  
https://doi.org/10.1016/j.jneumeth.2015.03.018

## Scientific Benchmarking

**Liu et al. (2025).**  
_LibEER: A Comprehensive Benchmark and Algorithm Library for EEG-Based Emotion Recognition._  
IEEE Transactions on Affective Computing.  
https://doi.org/10.1109/TAFFC.2025.3605833

Repository: https://github.com/XJTU-EEG/LibEER

**Kukhilava et al. (2025).**  
_Evaluation in EEG Emotion Recognition: State-of-the-Art Review and Unified Framework._  
https://arxiv.org/abs/2505.18175

Repository: https://github.com/EmotionLab/EEGain

**EEGEmoLib (2026).**  
_EEG-Based Emotion Recognition for Machine Intelligence: A Survey with an Open-Source Python Toolbox._  
National Science Review.  
https://doi.org/10.1093/nsr/nwag460

## Evaluation Bias and Generalization

**Moreno-Alcayde et al. (2024).**  
_Sneaky Emotions: Impact of Data Partitions in Affective Computing Experiments with Brain-Computer Interfacing._  
Biomedical Engineering Letters.  
https://doi.org/10.1007/s13534-023-00316-5

**Pandilova et al. (2026).**  
_Subject Identity Confounds qEEG Emotion Recognition on DEAP and DREAMER._  
Sensors.  
https://doi.org/10.3390/s26175327

**Gerster et al. (2026, preprint).**  
_Stimulus Identity Rather Than Emotion Drives EEG Classification on the FACED Dataset._  
bioRxiv.  
https://doi.org/10.64898/2026.06.12.731889

---

# 16. Final Project Direction

The primary goal is to create a **modular EEG emotion-recognition library that supports both dimensional emotional states and specific emotion categories**.

The software must provide practical functionality for preprocessing EEG, training recognition models, and producing predictions.

A central supporting objective is to ensure that the developed models can be evaluated rigorously and compared with existing scientific approaches.

The first research activities will focus on DEAP and SEED-IV, establishing classical and neural-network baselines for dimensional and specific emotion recognition.

Advanced feature representations, especially tensor decomposition, will then be investigated as potential improvements.

Finally, experiments on additional datasets will assess whether the proposed methods generalize beyond the original training participants and experimental conditions.

**The measure of success is a functioning emotion-recognition library supported by scientifically credible, reproducible evidence of its capabilities and limitations.**
