# Closing the Flower-to-Honey Pollen Domain Gap

Deep-learning pipeline for classifying pollen across two imaging domains — pollen collected
directly from flowers, and pollen recovered from honey via melissopalynological extraction —
and quantifying how badly models trained on one domain fail on the other. Code accompanying a
poster presented at **CEFood 2026** (Radisson Hotel, Plovdiv, Sept 23–26, 2026), with a full
manuscript to follow as a submission to *Food Science and Applied Biotechnology* (JFAB), UFT
Academic Publishing House, Plovdiv.

> **Status:** research code for a conference poster; the journal manuscript has not been
> submitted yet. See [Before you make this public](#before-you-make-this-public) below — there
> are a few things worth deciding first, mainly around sequencing this repo against the poster
> and the later JFAB submission.

## What this is

Five Bulgarian monofloral pollen classes — **Acacia, Brassica, Lavender, Thistle, Tilia** —
imaged in two domains under an identical protocol (glycerine-gelatine/fuchsin mounting, ZEISS
Primo Star, 40×):

- **Flower-collected pollen** — sampled directly from the flower.
- **Honey-recovered pollen** — extracted from honey via standard centrifugation
  (melissopalynological analysis).

Seven architectures are trained and evaluated under two conditions — **pure-flower** training
vs. **combined** (flower + honey) training — each with 10-fold cross-validation (140 training
runs total), then scored on two test sets that were never used in training:

| Test set | Size |
|---|---|
| Flower-domain (held out) | 765 images |
| Honey-domain (held out) | 45 images (9 per class) |

**Architectures:** a custom lightweight CNN, plus transfer-learned MobileNetV2, ResNet50,
ResNet152, InceptionResNetV2, Xception, VGG19.

## Headline result

Training only on flower-collected pollen gets **96.7–98.3%** accuracy on the flower-domain test
set, but only **25.1–69.8%** on the honey-domain test set (architecture-dependent) — a large,
previously unquantified domain gap. Adding honey-domain images to training (combined condition)
closes that gap to **90.9–99.6%** honey-domain accuracy, at effectively no cost to flower-domain
accuracy. The improvement is significant for all seven architectures (Welch's t-test, Holm-Bonferroni
corrected, p < .0001; Cohen's d 6.7–15.5, computed across 10-fold CV).

To the best of a literature search conducted for the manuscript, this is the first study to
directly quantify and close a flower-to-honey pollen domain gap — existing honey-pollen AI
classification work trains and tests within a single imaging domain.

## Repository structure

> **TODO — fill in to match your actual layout.** This session only ever saw
> `evaluate_all_models.py`; it does not know the real names or locations of your per-model
> training scripts, so don't publish this section as-is.

```
.
├── train_<model>.py          # TODO: one script per architecture, or one parameterized script?
├── evaluate_all_models.py    # rebuilds each architecture, loads fold_N.h5 checkpoints,
│                              # evaluates on both held-out test sets
├── data/                     # TODO: dataset location / how it's obtained (see Data below)
├── saved_models/
│   └── {condition}/{model}/fold_{1-10}.h5
├── results/
│   ├── final_evaluation_results.csv     # per-fold, per-model, per-condition test metrics
│   └── fold_level_results.csv           # see Known limitations — contains duplicate re-runs
├── figures/                  # TODO: analysis / chart scripts, if you're including them
└── README.md
```

## Setup

```bash
# TODO: confirm exact Python version
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Trained and evaluated with **TensorFlow/Keras 3.14.0**. TODO: pin exact package versions in
`requirements.txt` (TensorFlow, numpy, pandas, scikit-learn, h5py, whatever else the training
scripts import) and note the hardware used (GPU model, VRAM) — the manuscript currently has this
as an open `[Author: insert...]` placeholder too, so fill both from the same source.

## Usage

```bash
# TODO: real invocation once training-script names/args are confirmed
python train_<model>.py --condition pure_flower --fold 1

# Evaluate all trained checkpoints against both held-out test sets
python evaluate_all_models.py
```

`evaluate_all_models.py` rebuilds each architecture byte-identically to how it was trained,
loads the corresponding `fold_N.h5` checkpoint with `model.load_weights()`, and writes
`final_evaluation_results.csv`. This script and its output were independently re-verified this
project (bit-identical results under two different BatchNorm inference settings) — see
[Known limitations](#known-limitations) for what that check did and didn't resolve.

## Data

TODO — decide and state explicitly:

- Are the pollen microscopy images themselves being released, or is this a code-only repo?
- If released: under what license (code license ≠ image-dataset license — consider something
  like CC BY 4.0 for the images, separate from the code's MIT license below)?
- Any sampling/collection permissions or attributions (apiary, herbarium, institution) that need
  crediting?

If the dataset isn't being released, say so explicitly and describe the class/domain structure
(as above) so the results are at least interpretable without the raw data.

## Results

Full per-fold numbers are in `results/final_evaluation_results.csv`. Summary (honey-domain test
accuracy, mean across 10 folds):

| Model | Pure-flower training | Combined training | Gain |
|---|---|---|---|
| Custom CNN | 25.1% | 90.9% | +65.8 pts |
| ResNet50 | 51.8% | 97.3% | +45.6 pts |
| VGG19 | 56.7% | 99.6% | +42.9 pts |
| ResNet152 | 60.0% | 98.0% | +38.0 pts |
| Xception | 60.0% | 95.8% | +35.8 pts |
| MobileNetV2 | 65.1% | 97.6% | +32.4 pts |
| InceptionResNetV2 | 69.8% | 96.7% | +26.9 pts |

*(Sorted by pure-flower severity, worst first. TODO: double check these against the final CSV
before publishing — pulled from analysis done earlier in this project, not recomputed fresh for
this README.)*

## Known limitations

Be upfront about these rather than letting someone else find them:

- **`fold_level_results.csv` has an unresolved reproducibility issue.** It contains duplicate
  evaluation runs for 4 of the 6 pretrained models, and honey-domain accuracy (n=45) swings up to
  ~20 percentage points between repeated runs of the *same* model/condition/fold, while
  flower-domain accuracy (n=765) stays within ~2 points. This was investigated (a BatchNorm
  `training=True` vs. `False` hypothesis was tested and ruled out — see the manuscript's
  methodology notes) but the actual root cause is still unknown; it most likely lives in the
  original per-model training/evaluation notebooks rather than `evaluate_all_models.py`, which
  does not shuffle its evaluation batches and so cannot by itself reproduce that instability. If
  you're publishing this as "here's the code, reproduce our numbers," this needs a visible flag,
  not a quiet omission — someone will hit it.
- **`final_evaluation_results.csv` is independently verified**, separately from the point above:
  reloading every checkpoint fresh and re-evaluating under two different BatchNorm inference
  settings produced bit-identical results, both matching the manuscript's Table 1. That part is
  solid.
- The honey-domain test set is small (45 images, 9 per class) — one misclassified image moves
  accuracy by ~2.2 percentage points. Results near 100% should be read as indicative, not precise.

## Citation

Nothing is published yet, so there's no DOI to cite. Two things will exist in sequence and this
section should be updated as each one lands:

**1. Conference poster** (CEFood 2026, Sept 23–26, Plovdiv) — TODO: add the poster's formal
citation (title, authors, conference proceedings entry if CEFood publishes one) once it's fixed.

**2. Journal manuscript** (not yet submitted):

```
Zaykov, L. [et al.]. [Manuscript title — TBD, see note below]. Food Science and Applied
Biotechnology, [year], [vol]: [pages]. https://www.ijfsab.com
```

Note: the working title in the current manuscript draft foregrounds a "scale-preserving
preprocessing pipeline," but preprocessing was applied uniformly across all conditions and never
ablated — the training-data composition (pure-flower vs. combined) is what's actually
demonstrated to matter. Worth resolving the title before this citation block is finalized.

Also worth doing when the manuscript is actually submitted: JFAB will likely ask (directly or via
a cover-letter originality declaration) whether this work has been previously presented or
disseminated. It has — at CEFood 2026, and via this repo. Disclose both rather than letting a
reviewer discover the poster or repo independently.

## License

Code: [MIT](LICENSE).

This does **not** automatically cover the pollen image dataset (see Data, above — pick a
separate license for that if you release it) or the manuscript/poster text and figures, which
are typically bound by whatever copyright/licensing agreement you sign with JFAB/UFT Academic
Publishing House upon acceptance — check that agreement (including its self-archiving / green-OA
policy) before including manuscript PDFs or poster files in this repo.

## Before you make this public

A few things worth deciding *before* you hit "Create repository," not after:

1. **Prior-dissemination disclosure, not blind review.** The poster is presented publicly under
   your name at CEFood 2026, so there's no anonymity to protect at the repo stage — a public
   GitHub repo doesn't deanonymize anything that the poster itself hasn't already made public.
   The actual thing to get right is the reverse: when you submit to JFAB later, disclose that
   this work was previously presented at CEFood 2026 and that the code is public. Many journals
   ask this directly (originality/prior-publication declaration); getting caught not disclosing
   it is worse than disclosing it.
2. **Sequencing.** Since the poster comes first, it makes sense to hold the repo's "About"
   description and this README to match what the poster actually claims at the time it's
   presented (Sept 23–26) — don't let the repo describe results or framing that the poster
   itself doesn't yet support, and update both together if the numbers change between now and
   the conference.
3. **Copyright/self-archiving, for later.** Once you do submit to JFAB and (if) it's accepted,
   most publishers require either a copyright transfer or an exclusive/non-exclusive license, and
   many restrict what you can host publicly and when (some require an embargo). MIT-licensing
   your code doesn't clear you to post the manuscript itself — that's a separate decision to make
   at submission/acceptance time, not now.
4. **The `fold_level_results.csv` instability is still open.** Either resolve it before release,
   or make sure the Known Limitations section above (or an equivalent) actually ships with the
   repo — don't let this be the first thing a reader finds on their own, especially with a
   conference audience about to look at the same numbers.
5. **Tie this back to the manuscript, later.** When you do write the manuscript, its Data
   Availability placeholder should point at this repo (and this repo should note the eventual
   DOI/citation) — keep the two in sync rather than letting them drift.
