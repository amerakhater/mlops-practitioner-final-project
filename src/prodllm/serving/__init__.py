"""The BentoML serving layer.

Kept in its own package, and out of ``prodllm.api``, because the two serving
surfaces have different jobs and different dependencies:

``prodllm.api``      FastAPI. The embedding and metadata contract from Module 1,
                     published as openapi.json and gated in CI. It imports no
                     bentoml, so the runtime image stays thin.
``prodllm.serving``  BentoML. The Module 3 production surface: an asynchronous
                     ``/ask``, token streaming, batch re-indexing and the
                     canary-addressable release identity.

Both drive the same ``prodllm.rag.RagEngine``. Retrieval, citation and refusal
behaviour is therefore defined once, and the one existing test suite still
covers it for both.
"""
