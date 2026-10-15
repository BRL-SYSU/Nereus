# Research code release

Released on 2026-10-15 from the existing AquaCount-instruct experiments.
The original experiment workspace was not modified.

Included: canonical data preparation utilities; P2PNet baseline and compatibility
patch; external CountGD/CLIP-Count adapters; closed-VLM inference and metrics;
count-range post-processing; accompanying tests and protocol notes.

Publication changes are limited to replacing private server paths with repository
relative paths, updating the model-root path assertion, and clarifying the public
setup instructions. Run commands from the repository root. Evaluation algorithms
and frozen numeric protocol constants are unchanged.

Excluded: datasets, model weights, external source trees, environments, private
API configuration, prediction caches, logs, and local launch-session scripts.
No claim is made that this subset reproduces all tables in the paper.

Validation: 25 independent CPU metric/protocol tests and 4 count-range tests
passed; all Python files compiled. The prepare-data integration test requires
the real frozen split and annotations and is not part of that standalone suite.
GPU training, model inference and paid API requests were not rerun for release.

External source licenses remain applicable; the P2PNet compatibility patch does
not supersede upstream terms. No new blanket software license is assigned.
