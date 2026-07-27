"""Dataset adapters — convert other tracking datasets into the SA-Co ``_ext`` JSON schema.

The pipeline's abstraction boundary is the annotation-JSON schema (the loader and the vendored VEval both
bind to it), so a dataset is added by emitting that schema — not by touching the loader. R7 (independent
replication) uses :mod:`src.adapters.mammalps`; box-only datasets go through :mod:`src.adapters.boxes`
(box → filled-rectangle RLE) with ``eval.prefer_bbox`` for box-HOTA.
"""
