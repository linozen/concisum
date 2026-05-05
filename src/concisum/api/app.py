from __future__ import annotations

import logging
import os

from litestar import Litestar, get
from litestar.di import Provide

from concisum.api.controllers import JobController
from concisum.api.job_store import JobStore
from concisum.api.pipeline_controller import PipelineController

LOG = logging.getLogger(__name__)

_job_store: JobStore | None = None


def provide_job_store() -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store


@get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


def create_app() -> Litestar:
    listen = os.getenv("LISTEN_ADDR", ":8090")
    LOG.info("concisum API configured on %s", listen)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    return Litestar(
        route_handlers=[health_check, JobController, PipelineController],
        dependencies={"job_store": Provide(provide_job_store, sync_to_thread=False)},
        debug=True,
    )


app = create_app()
