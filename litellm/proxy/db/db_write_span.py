"""A ``ServiceTypes.DB`` event around a Prisma write that ``@log_db_metrics`` cannot wrap.

The spend flush and the spend-log batch insert run raw ``prisma_client.db``
transactions inside retry loops, so the decorator (one event per decorated
coroutine) cannot name the table each transaction touches. This context manager
emits one success or failure event per transaction, carrying the raw ``call_type``
for the metric labels and the Prisma model on ``table_name`` so OTel renders
``postgres.{verb} {table}``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm._service_logger import ServiceTypes
from litellm.proxy.db.log_db_metrics import _is_exception_related_to_db


async def _emit_failure(
    call_type: str, event_metadata: Mapping[str, str], start_time: datetime, error: Exception
) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    end_time: Final = datetime.now()
    try:
        await proxy_logging_obj.service_logging_obj.async_service_failure_hook(
            error=error,
            service=ServiceTypes.DB,
            call_type=call_type,
            parent_otel_span=None,
            duration=(end_time - start_time).total_seconds(),
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
        )
    except Exception as hook_error:
        verbose_proxy_logger.debug("db_write_span: failure hook raised for %s: %s", call_type, hook_error)


@asynccontextmanager
async def db_write_span(call_type: str, table: str) -> AsyncGenerator[None]:
    from litellm.proxy.proxy_server import proxy_logging_obj

    start_time: Final = datetime.now()
    event_metadata: Final = {"table_name": table}
    try:
        yield
    except Exception as e:
        if _is_exception_related_to_db(e):
            await _emit_failure(call_type, event_metadata, start_time, e)
        raise
    else:
        end_time_ok: Final = datetime.now()
        asyncio.create_task(
            proxy_logging_obj.service_logging_obj.async_service_success_hook(
                service=ServiceTypes.DB,
                call_type=call_type,
                parent_otel_span=None,
                duration=(end_time_ok - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time_ok,
                event_metadata=event_metadata,
            )
        )
