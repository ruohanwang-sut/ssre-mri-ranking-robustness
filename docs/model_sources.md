# Original data and third-party model sources

Original fastMRI data must be obtained from https://fastmri.med.nyu.edu/ under
the provider's application and data-use terms. Original H5 files are not redistributed.

Official software: https://github.com/facebookresearch/fastMRI,
https://github.com/hellopipu/PromptMR,
https://github.com/hellopipu/PromptMR-plus.
Use the providers' documented brain-checkpoint download instructions. Weights
are not redistributed. Verify downloaded weights against docs/frozen_configuration.json.
The checkpoint identities and official repository commits are frozen study records.
They are not a claim that the provider's current default branch has the same behavior.

Full model inference requires externally acquired MRI data, the matching official
repositories and weights, and the documented environment. The archive scripts are
historical source with original local paths and dependencies; they are not a turnkey
portable runner. The maintained public entry point reproduces statistics from CSV,
and does not start training or rerun the 68-volume GPU experiment.
