# Parkinson's Disease Classification from Acoustic Voice Features

This project explores whether acoustic measures of vocal production can reliably distinguish individuals with Parkinson's disease from healthy controls, and more specifically, which features carry the strongest diagnostic signal.

The dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/174/parkinsons) and contains 195 voice recordings from 31 individuals (23 with Parkinson's, 8 healthy). Each recording is summarized by 22 acoustic features capturing irregularities in pitch, amplitude, and vocal stability that are known to emerge in Parkinson's-related dysarthria.

---

## Why This Question

Parkinson's disease affects the motor system, and the voice is one of the earliest and most measurable places that disruption shows up. Subtle changes in how someone controls their vocal folds, sustains a pitch, or manages breath support can appear years before a clinical diagnosis. That makes acoustic analysis a compelling candidate for non-invasive, scalable biomarker development.

The clinical challenge is that many of these changes are imperceptible to the human ear in early stages. Computational approaches can quantify them. This project is a small exploration of how well machine learning handles that classification task, and what the feature importances suggest about which aspects of vocal motor control are most disrupted.

---

## Dataset Features

The 22 acoustic features broadly fall into four categories:

| Category | Features | What they capture |
|---|---|---|
| Fundamental frequency | MDVP:Fo, Fhi, Flo | Average, max, and min pitch. Instability here reflects reduced laryngeal motor control |
| Jitter measures | MDVP:Jitter(%), MDVP:Jitter(Abs), RAP, PPQ, DDP | Cycle-to-cycle variation in pitch. A marker of vocal fold irregularity |
| Shimmer measures | MDVP:Shimmer, Shimmer(dB), APQ3, APQ5, APQ11, DDA | Cycle-to-cycle variation in amplitude. Reflects breath support and vocal fold mass changes |
| Noise and complexity | HNR, NHR, RPDE, DFA, PPE, spread1, spread2, D2 | Harmonics-to-noise ratio, signal complexity, and nonlinear dynamical measures |

The target variable `status` is binary: 1 = Parkinson's, 0 = Healthy.

---

## What the Data Shows Before Any Modeling

A few things stand out just from the distributions.

Fundamental frequency (Fo) is noticeably lower and more tightly clustered in Parkinson's speakers compared to healthy controls, where it spreads more broadly. This likely reflects reduced range of laryngeal movement.

Jitter and shimmer are both elevated in Parkinson's. The distributions shift right and widen. The overlap is substantial, but the central tendency is meaningfully different, consistent with what the literature describes as increased vocal instability due to impaired motor control.

HNR (Harmonics-to-Noise Ratio) is lower in Parkinson's, meaning the signal is noisier relative to the harmonic component. Healthy voices skew toward higher values. This is one of the cleaner separations visually.

RPDE and PPE, both nonlinear complexity measures, show the most visually distinct separation between groups. These features do not map as intuitively to vocal anatomy, but they appear to capture something about the regularity of phonation that the simpler measures miss.

The correlation matrix reveals strong internal clustering: jitter measures correlate tightly with each other, shimmer measures do the same, and the nonlinear features form their own cluster. This redundancy is worth keeping in mind when interpreting feature importances, since any one feature from a correlated cluster could stand in for the others.

---

## Modeling Approach

Four classifiers were evaluated using stratified 5-fold cross-validation, with AUC as the primary metric given the class imbalance (roughly 75% Parkinson's):

| Model | Mean AUC (CV) |
|---|---|
| Logistic Regression | 0.899 +/- 0.037 |
| SVM (RBF) | 0.891 +/- 0.044 |
| Random Forest | 0.955 +/- 0.021 |
| Gradient Boosting | 0.960 +/- 0.014 |

Random Forest was selected for test set evaluation given its lower variance and strong CV performance. On the held-out test set it achieved:

- Accuracy: 92%
- ROC-AUC: 0.962
- Precision on Parkinson's class: 0.93, Recall: 0.97
- Only 1 false negative (Parkinson's classified as healthy) across 39 test samples

False negatives are the clinically costly error here, so the low false negative rate is the more meaningful result.

---

## Feature Importances

Permutation importance was used rather than Gini-based importance, which tends to inflate the apparent relevance of high-cardinality continuous features. The top contributors were:

1. **MDVP:Fhi(Hz)** - Maximum fundamental frequency. Its high importance suggests that the upper range of pitch control is more disrupted in Parkinson's than the average pitch alone.

2. **PPE (Pitch Period Entropy)** - Measures irregularity of fundamental frequency across a recording. High PPE indicates difficulty sustaining a stable pitch, consistent with Parkinson's-related loss of fine laryngeal motor control.

3. **MDVP:Fo(Hz)** - Average fundamental frequency. Lower and less variable in Parkinson's speakers, reflecting reduced range of laryngeal movement.

4. **spread1** - A nonlinear measure of fundamental frequency variation. Highly correlated with PPE and similarly sensitive to dysregulation in pitch control.

5. **MDVP:Flo(Hz)** - Minimum fundamental frequency. Together with Fhi, captures the total pitch range and overall laryngeal motor flexibility.

---

## Limitations and Open Questions

The dataset is small (195 recordings, 31 speakers) and imbalanced. The strong AUC results should be interpreted with caution since high performance on a small, clean dataset does not guarantee generalization to clinical populations.

A few things this analysis cannot answer:

- How do these features change longitudinally as Parkinson's progresses?
- Can the same features distinguish Parkinson's from other neurodegenerative conditions with overlapping speech symptoms such as ALS or PSP?
- Are the top features stable across recording conditions, microphone types, and languages?

These are the questions that make this an interesting research problem rather than a closed one.

---

## Running the Project

```bash
git clone https://github.com/tanvisanjeev/parkinsons-speech-biomarkers.git
cd parkinsons-speech-biomarkers
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 parkinsons_classification.py
```

All plots are saved automatically to the project folder.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## Data Source

Little, M., McSharry, P., Roberts, S., Costello, D., and Moroz, I. (2007).
Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection.
BioMedical Engineering OnLine, 6(23).
UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/174/parkinsons