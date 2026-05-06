**LAB TO FIELD BENCHMARK \- RESEARCH PLAN**

**Key Papers**

**VERIFIED RESEARCH PAPERS \- DIRECTLY RELEVANT TO YOUR BENCHMARK**

**1\. Wu et al. (2023) \- From Laboratory to Field: UDA for Plant Disease Recognition in the Wild**  
closest prior work must cite  
Problem: Models trained on Plant Village (lab) degrade sharply when tested on field datasets. Method:  
Proposes MSUN — a multi-representation subdomain adaptation network with uncertainty regularization and local maximum mean discrepancy (LMMD) to align cross-species feature distributions. Uses a non-adversarial approach. Datasets: Plant Village (source), PlantDoc, Plant-Pathology, Corn-Leaf-Diseases, Tomato-Leaf-Diseases (targets). Gap it leaves: Tests one specific UDA method; no standardized protocol to compare multiple adaptation strategies side by side. No domain gap quantification metric. Your benchmark fills this directly.  
Published: Plant Phenomics, 2023\. DOI: 10.34133/plantphenomics.0038  
**2\. Tunio et al. (2024) \- Transformer-Fused Convolution \+ Wasserstein Domain Adaptation key baseline** adaptation method  
Problem: Standard CNNs on Plant Village reach 99.35% but fall to 31.4% on field environments — a catastrophic 68-point drop. Method: Combines transformer and CNN in an NFRegNet architecture with Wasserstein distance-based domain alignment to reduce distributional shift. Datasets: Plant Village, PlantDoc, plus an augmented in-house dataset. Gap it leaves: Still evaluated on only two datasets; no systematic ablation of the adaptation contribution vs. architectural contribution. Your benchmark would isolate these effects across more datasets.  
Published: Computers & Electronics in Agriculture, 2024\. DOI: 10.1016/j.compag.2024.109574  
**3\. Yang et al. (2024) — Cross-Domain Few-Shot Learning for Crop Disease in the Field few-shot angle**  
directly relevant  
Problem: FSL methods trained on Plant Village fail on field datasets because they assume source and target share the same domain, which is violated in real conditions. Method: Proposes a CD-FSL (cross-domain few-shot learning) framework that explores inter-domain correlations from both data and model perspectives. Datasets: PlantVillage (source), multiple field crop datasets. Gap it leaves:  
Focused only on FSL settings; does not benchmark standard fine-tuning, domain adversarial, or zero-shot methods under the same protocol. Your benchmark should include FSL as one row in a broader comparison table.  
| Published: Frontiers in Plant Science, 2024\. DOI: 10.3389/fpls.2024.1434222  
**4\. Wei et al. 2024\) \- PlantWild: Benchmarking In-the-Wild Plant Disease Recognition (ACM MM**  
**2024\)**  
closest benchmark paper  
differentiate from this  
Problem: No large-scale wild plant disease dataset with diverse disease classes exists for proper benchmarking. Method: Curates PlantWild (18,542 images, 89 classes) and proposes MVPDR, a multimodal baseline using text descriptions and visual data. Benchmarks multiple methods including few-shot and training-free settings. Gap it leaves: PlantWild is multimodal (image+text); your benchmark is RGB-only, which is more deployment-realistic. Also, PlantWild does not measure domain gap with formal metrics, does not compare adaptation strategies systematically, and does not define a domain gap score. These are your differentiating contributions.  
Published: ACM Multimedia 2024\. arXiv: 2408.03120. Dataset on HuggingFace.  
**5\. Chalvatzaki et al. (2025) \- Bridging Lab-to-Field Gap via UDA \+ Background Recomposition (ScienceDirect)**  
novel method, 2025  
Problem: Background clutter in field images is the primary source of lab-to-field degradation, not disease appearance itself. Method: Uses background recomposition to synthesize training images that mimic field backgrounds, then applies UDA. Gap it leaves: Tests one mitigation strategy (background augmentation); your benchmark can treat this as one of several adaptation conditions and compare it systematically against DANN, fine-tuning, and others. Confirms background shift is a measurable and isolatable component of domain gap.  
| Published: Ecological Informatics, 2025\. DOI: 10.1016/j.ecoinf.2025  
**6\. Plenk et al. (2025) \- Benchmarking Transfer Learning on Open Datasets for Plant Disease (Scientific Reports)**  
benchmark study  
structurally similar  
Problem: No systematic comparison of transfer learning strategies across multiple plant disease datasets exists. Method: Evaluates domain-specific vs. ImageNet pretrained models across multiple open datasets (PlantDoc, Plant Village, novelPotato, Taiwan Tomato). Notes that PlantDoc achieves comparatively very low accuracy for most models. Gap it leaves: Compares transfer learning backbones but not domain adaptation methods; no domain gap metric is defined; no staged adaptation protocol.  
Your paper extends this direction with adaptation methods and a formal gap quantification.  
Published: Scientific Reports, 2025\. DOI: 10.1038/s41598-025-03235-w  
**7\. Ramirez et al. (2024) \- BioRxiv: Domain Shift in Plant Disease Diagnosis and Flower Recognition**  
direct evidence  
Problem: Directly investigates domain shift in plant disease detection across architectures. Key finding:  
Performance degradation from lab to field is consistent across all model architectures, suggesting that domain shift cannot be addressed through model architecture alone — dataset diversity is essential.  
Gap it leaves: Diagnostic only; proposes no adaptation protocol; uses a limited set of datasets; no benchmark structure for future comparisons.  
| Preprint: bioRxiv, October 2024\. DOI: 10.1101/2024.10.07.617111

**Pipeline**

**END-TO-END ARCHITECTURAL PIPELINE**

**1\. Dataset selection and domain tiering**  
Organise your datasets into three tiers based on acquisition conditions. Tier 1 (Lab): Plant Village — uniform backgrounds, controlled illumination, single-leaf images. This is your source domain in all experiments. Tier 2 (Semi-field): PlantDoc, Plant-Pathology 2021 (Kaggle) — internet-scraped, mixed conditions, varied backgrounds. Tier 3 (Wild): PlantWild, Cassava Leaf Disease, any field-collected crop-specific dataset. Tiering lets you measure degradation in stages, not just a single lab-vs-field jump. The cassava dataset adds a cross-crop dimension, which is important for generalization claims.  
Plant Village  
PlantDoc  
PlantWild  
Cassava  
**2\. Class alignment and subset construction**  
Not all datasets share the same disease classes. You must build aligned class subsets for cross-dataset evaluation. Strategy: identify overlapping disease categories across at least two datasets (e.g., tomato early blight appears in Plant Village, PlantDoc, and PlantWild). Build a shared class vocabulary of N diseases with matched labels across source and target. For unmatched classes, use them only in within-dataset evaluations. Document every alignment decision — this is part of your benchmark contribution.  
**3\. Standardised preprocessing protocol**  
One of your contributions is a reproducible protocol. Define: image resize (224×224), normalisation (ImageNet mean/std), and a fixed augmentation policy for training only (horizontal flip, colour jitter). Do NOT use  
background removal or segmentation as preprocessing — this would eliminate the domain shift you are trying to measure. Keep all images in their natural state. Apply identical preprocessing to all datasets so that the only variable is the domain, not the preprocessing.  
**4\. Backbone selection and frozen feature extraction**  
Use two backbone families: a CNN (EfficientNet-BO or ResNet-50) and a transformer (Swin-T or DeiT-Small). Both pretrained on ImageNet only — no plant-specific pretraining at this stage. Freeze the backbone and extract feature embeddings from the penultimate layer for all images across all datasets.  
These frozen embeddings are used for: (a) computing domain gap metrics before any adaptation, and  
(b) as a starting point for adaptation methods / two-backbone design lets you separate architectural effects from adaptation effects.  
**5\. Baseline: source-only training (no adaptation)**  
Train the classifier on PlantVillage only. Evaluate on all target datasets without any adaptation. Record accuracy, macro-F1, and per-class F1 on every target dataset. This establishes your "no adaptation" floor and demonstrates the domain gap quantitatively. This single experiment produces the core finding that motivates your entire benchmark — the degradation table.  
**6\. Adaptation strategy bank**  
Implement 5-6 adaptation strategies as separate conditions. Each is trained under the same protocol on the same source domain, then evaluated on the same target domains. Strategies to include: (A)  
Source-only (baseline, no adaptation). (B) Fine-tuning on labelled target data (supervised upper bound). (C) Data augmentation only — aggressive background augmentation, colour jitter, cutmix. (D)  
Domain adversarial training (DANN). (E) Contrastive pretraining on unlabelled target images (SimCLR-based SSL), then fine-tune. (F) Wasserstein or MMD-based distribution alignment. Each of these represents a category of adaptation technique, and your benchmark systematically compares them under a fixed evaluation protocol \- that is the contribution.  
Source-only  
Fine-tune  
Augment  
DANN  
SSL+FT  
MMD/Wasserstein  
**7\. Cross-dataset evaluation protocol**  
Every adaptation strategy is evaluated on all three domain tiers. Run each strategy with 3 random seeds and report mean \+ standard deviation. Evaluation splits: use the official test split of each target dataset, or a fixed 20% stratified split if no official split exists. Never use target data for training in UDA conditions; this is a strict requirement for scientific validity. For supervised fine-tuning conditions, use 10%, 20%, and 50% of labelled target data to show a label-efficiency curve.  
**8\. Ablation and component analysis**  
For each adaptation method that improves on baseline, run ablations to isolate which component contributes. For SSL+FT: compare with and without the pretraining step. For DANN: compare with and without the gradient reversal layer. For augmentation: compare with and without each augmentation type. Ablations transform your benchmark from a leaderboard into a scientific analysis — they answer not just "what works" but "why it works."

**Benchmark Pipeline**

**BENCHMARK STRUCTURE AND EVALUATION DESIGN**

**1\. Benchmark framing — what makes this different from prior work**  
Your benchmark is the first to: (1) define a three-tier domain taxonomy for plant disease datasets (lab / semi-field / wild), (2) evaluate multiple adaptation strategies under a single fixed protocol across all tiers simultaneously, (3) report a formal domain gap score (not just accuracy) as a first-class metric, and  
(4) separate architectural effects from adaptation effects by fixing the backbone. PlantWild (Wei et al.,  
2024\) benchmarks classification methods but uses multimodal data and no adaptation methods. Plenk et al. (2025) tests transfer learning backbones but no adaptation strategies. Your work covers the space between these two papers.  
**2\. Evaluation conditions matrix**  
The core output of your benchmark is a matrix: rows are adaptation strategies (A through F), columns are target datasets (PlantDoc, PlantWild, Cassava, others). Each cell contains macro-F1 and accuracy.  
Additionally, a separate column reports the domain gap score for each source »target pair. This matrix is the central table in your paper — it shows at a glance which strategy generalises best to which type of field condition.  
Important: all strategies must share the same source domain (PlantVillage), same backbone, same preprocessing, and same evaluation splits. The only variable that changes between rows is the adaptation strategy. This controlled design is what makes the comparison scientifically valid.  
**3\. Dataset splits \- preventing data leakage**  
Use official train/test splits where they exist. For Plant Village, the Hugging Face version provides leaf-group-aware splits that prevent the same leaf appearing in train and test. For target datasets used in UDA settings, training images are used unlabelled (for methods that require target images during training); test images are always labelled and held out. For supervised fine-tuning, randomly sample labelled subsets from the training split only. Document all split decisions in a reproducibility appendix.  
**4\. Baseline performance expectations**  
Based on literature, expect the source-only condition to achieve: \~95-99% on Plant Village (in-domain), \~55-70% on PlantDoc, \~45-60% on PlantWild, \~60-70% on Cassava. The best adaptation method should improve field performance to \~72-85%. The supervised fine-tuning upper bound (100% target labels) sets the ceiling. The gap between source-only and supervised fine-tuning is what your adaptation methods are trying to close — and your contribution is showing how much of that gap each strategy recovers.

**GAP METRICS**

**DOMAIN GAP METRICS \- THE NOVEL QUANTIFICATION CONTRIBUTION**

| Domain Gap Score (DGS) Primary metric. Defined as: DGS \= Acc(source→source) \- Acc(source target). A higher DGS means a larger gap. Report this for each source target pair and each adaptation strategy. A good adaptation method should reduce DGS without requiring labelled target data. | Macro-F1 drop (AF1) More informative than accuracy alone for imbalanced datasets. Compute AF1 \= F1(source→source) \- F1(source →target). Report per-class AF1 to identify which disease categories suffer most from domain shift \- a finding with direct agronomic implications. |
| :---- | :---- |
| **Maximum Mean Discrepancy (MMD)** Measures feature-space distributional distance between source and target embeddings. Computed on frozen backbone features before adaptation. A lower MMD after adaptation indicates the method is successfully aligning distributions. Use as a proxy for how hard a particular source →target pair is. | **Label Efficiency Curve** For supervised fine-tuning: plot accuracy vs. percentage of labelled target data (5%, 10%, 20%, 50%, 100%). Shows how quickly each adaptation method recovers performance as labels are added. This is directly useful for practitioners choosing how much labelling effort to invest. |
| **Adaptation Gain (AG)** AG \= Acc(adapted) \- Acc(source-only) on the target set. Normalised AG \= AG / (Acc(supervised upper bound) \- Acc(source-only)). Normalised AG tells you what fraction of the possible improvement a given unsupervised method achieves. This is your clean headline metric. | **Background Sensitivity Index (BSI)** Novel metric specific to your work. Compute accuracy on Plant Village with segmented (black background) vs. colour (natural) versions of the same images. BSI \= drop due to background presence. Decompose domain gap into background contribution and disease-appearance contribution. This isolates the background shift, as suggested by Chalvatzaki et al. (2025). |

**How to report the domain gap formally**  
For each source-target pair, report a row in a gap table: dataset name, MMD score (distributional gap), DGS (accuracy-based gap), AF1 (macro F1 gap), BSI (background contribution). Rank dataset pairs by MMD to show which transitions are hardest. This gives your benchmark a diagnostic tool that practitioners can use to predict how hard a new field dataset will be before running experiments.

**EXPECTED OUTCOMES**

**EXPECTED OUTPUTS AND CONTRIBUTIONS**

**1\. Primary deliverable — the benchmark results table**  
A full matrix of 6 adaptation strategies × 3+ target datasets with accuracy, macro-F1, normalised adaptation gain, and MMD. This is Table 1 in your paper. Every row is reproducible under your fixed protocol. This table alone is a publishable contribution at IEEE Access, Computers & Electronics in Agriculture, or Frontiers in Plant Science.  
**2\. Secondary deliverable \- domain gap characterisation**  
The domain gap table (DGS, MMD, AF1, BSI per dataset pair) gives the field a diagnostic tool that did not exist before. It lets future researchers pick the right adaptation strategy for their specific source-target transition without running all experiments themselves. This is presented as a practical guide in a dedicated section of your paper.  
**3\. Expected key findings (hypotheses to confirm)**  
Based on literature: (1) Source-only training degrades catastrophically on all field datasets. (2) Fine-tuning with even 10-20% target labels recovers most performance — the label efficiency result. (3) DANN and MMD-based methods help but do not close the full gap without labels. (4) Background augmentation alone closes a meaningful portion of the gap (consistent with Chalvatzaki 2025). (5) Transformer backbones (Swin-T) degrade less than CNNs under domain shift — consistent with PMC  
2025 benchmark showing Swin at 88% vs CNNs at 53% on real-world data.  
**4\. Code and reproducibility artefacts**  
Release: a GitHub repository with dataset loading scripts, preprocessing pipeline, all adaptation method implementations, evaluation scripts, and the full results table as a CSV. Reproducibility is a publishability criterion at top venues — a GitHub repo with clean code substantially improves acceptance odds at journals like IEEE Access and Frontiers.

---

**On the papers:** The most critical paper to understand deeply is Wu et al. (2023) in Plant Phenomics, because it is the single closest predecessor to your idea — same problem, same source domain, but it proposes one specific UDA method rather than benchmarking multiple strategies. Plant Village is consistently used as the source domain in all experiments since it is collected in the laboratory environment with a plain background Science, and this is precisely the convention your benchmark should standardise. The Tunio et al. (2024) paper provides your most important numerical anchor: despite demonstrating an impressive accuracy of 99.35% on the Plant Village test set, the recognition accuracy experienced a significant decline to 31.4% when confronted with field environments ScienceDirect \- this single statistic is the motivating evidence for your entire paper.  
**On what makes your benchmark novel relative to Plant Wild:** Plant Wild imposes a plethora of challenges including accurate identification of disease areas from complex backgrounds, large intra-class appearance variances and small inter-class discrepancies arXiv \- but their benchmark tests classification methods, not adaptation strategies, and their dataset is multimodal (text \+ image). Your benchmark is RGB-only (deployment realistic), systematically compares adaptation strategies, and introduces formal domain gap scoring. These are non-overlapping contributions.  
**On the domain gap metric rationale:** The literature consistently treats accuracy drop as the only measure of domain shift, but this conflates model quality with dataset difficulty. By proposing MMD on frozen features as a dataset-level difficulty metric (independent of the classifier), you give the field something it currently lacks — a way to predict how hard a new field dataset will be before running experiments. Performance degradation was consistent across all model architectures, suggesting that domain shift cannot be addressed through model architecture alone bioRxiv, which justifies measuring gap at the data level, not the model level.  
**On the uncertainty extension:** This is a Phase 1 add because Wu et al. already uses uncertainty regularisation during training \- your contribution is measuring it as an output metric rather than a training signal. It adds one ECE column to your results table and one figure, with minimal extra implementation effort.

---

**What I have done till now :** 

The Idea 4 project focuses on studying domain shift in crop disease detection by evaluating how a model trained on controlled lab images (PlantVillage) performs on real-world images (PlantDoc). I aligned both datasets by cleaning class names and creating a mapping to obtain 17 common classes, followed by building a unified dataset structure with consistent train/validation splits. Using this, I trained a baseline EfficientNet-B0 model on PlantVillage, achieving high validation accuracy (\~85–97%). However, cross-domain evaluation showed a sharp drop on PlantDoc (\~21–26% accuracy, \~0.16–0.20 macro-F1), confirming a significant domain gap. This was further supported by feature-level metrics such as MMD² (\~0.12–0.14), centroid distance (\~0.37–0.44), and cosine similarity (\~0.75–0.86), indicating a clear distribution mismatch between domains.

To improve generalization, I introduced stronger domain-aligned augmentations and a two-phase training strategy, which reduced overfitting to the source domain (PlantVillage accuracy \~85–88%) but did not significantly improve PlantDoc performance. I then performed supervised fine-tuning on the target domain, which substantially improved PlantDoc performance (\~49–54% accuracy, \~0.43–0.49 macro-F1) while maintaining strong performance on PlantVillage (\~86–93%). This reduced the domain gap (accuracy drop reduced from \~73–75% to \~40–45%) and also decreased the feature-level gap (MMD² reduced from \~0.13 to \~0.08). Despite this improvement, a notable gap still remains, indicating that the performance difference is primarily due to distribution mismatch between lab and real-world images, motivating the need for more advanced domain adaptation techniques.

