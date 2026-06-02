"""MARS: Fast Inference for Augmented LLMs.

vLLM **v1** extension layer. This package is a clean re-port of the MARS
research system (originally a fork of vLLM 0.2.0) onto modern vLLM, built as an
extension on top of an *unmodified* vLLM via supported injection points
(``scheduler_cls``, ``SamplingParams.extra_args``, native resumable streaming,
and the KV-connector interface) rather than by forking the engine.

See the implementation plan for the phased design.
"""

__version__ = "0.0.1"
