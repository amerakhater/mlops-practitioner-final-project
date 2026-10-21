
import logging
from fastapi import FastAPI

from prodllm.logging_conf import configure_logging
from prodllm.api.middleware import CorrelationIdMiddleware

configure_logging(level=logging.INFO)

app = FastAPI()
app.add_middleware(CorrelationIdMiddleware)
