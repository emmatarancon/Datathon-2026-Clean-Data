"""rag2: measured retrieval improvements for the Math on the Edge copilot.

Three techniques, each activated only if dev measurements prove it useful:
  gate   - similarity threshold + latency budget (rag2/gate.py)
  hybrid - BM25 + cosine fusion (rag2/hybrid.py)
  dedup  - answer transfer across near-duplicate problems (rag2/dedup.py)

Standard library + copilot imports only (board rule R4). The only change to
existing files is the minimal hook in copilot/pipeline.py plus the RAG_MODE
keys in config.env; with RAG_MODE=baseline nothing here runs at all.
"""
