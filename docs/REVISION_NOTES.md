# Revision: integrity, interpretation and presentation

This revision preserves the completed model runs, original scores and training protocol. It improves the software's handling of cached evidence and the explanation of the experiment.

## Software changes

Resumed primary inference now verifies source RGB/GT hashes and cached mask dimensions and train IDs. Newly written predictions also receive SHA256 hashes. When those prediction hashes are present, both resume and evaluation verify them. Older measured manifests predate prediction hashes: their source provenance and mask validity can be checked, but this revision cannot retroactively prove that an old prediction file never changed.

Six regression tests cover changed source inputs, altered predictions, invalid cached labels, missing prediction files and a mismatched ground-truth manifest hash and preservation of run metadata on failed validation. Model arithmetic, checkpoint files, data partitions, optimization and saved experimental measurements remain as originally executed. The pinned source-only checkout in the notebook reproduces the original experiment implementation; current main additionally contains the integrity improvements described here.

## Report and presentation

The report explains why aggregate gains can coexist with poor rider performance and distinguishes the released UDA comparison from the freshly trained target-supervised DINOv3 head. The revised presentation has separate tuning and full-refit slides, visible uncertainty beside the primary finding, and a focused sparse-class failure slide. Source citations and speaker notes accompany the claims.

## Additional analysis

An exploratory recording-family omission analysis uses saved confusion counts, without rerunning inference or changing the original hypothesis. See `RECORDING_SENSITIVITY.md`. It measures sensitivity to the observed groups, rather than providing an independent test set or a confidence interval.

## Authorship and disclosure

AI assistance is disclosed in the repository and report. No external similarity or AI-detector score is claimed. The report's evidence, references and limitations should be understood and explained by the submitting student in accordance with course rules.
