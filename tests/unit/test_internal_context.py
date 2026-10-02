"""``with_service_target`` and ``service_caller`` carry the purpose and the caller of a datastore call
to code that cannot see them from its own frames, and every Redis producer on the proxy request path
declares a key family so no request-path span renders as a bare ``redis.get``."""

import asyncio
import contextvars
import re
from collections.abc import Generator
from pathlib import Path
from typing import Final

import pytest

from litellm._internal_context import (
    current_service_caller,
    current_service_target,
    service_caller,
    service_target,
    with_service_target,
)

_REPO: Final = Path(__file__).resolve().parents[2]

_REQUEST_PATH_PRODUCER_DIRS: Final = ("litellm/proxy/hooks", "litellm/router_strategy")

_CACHE_CALL: Final = re.compile(
    r"\.(?:async_)?(?:get_cache|set_cache|batch_get_cache|batch_get_cache_shared|increment_cache|increment"
    r"|set_cache_pipeline|set_cache_pipeline_with_ttls|set_cache_sadd|delete_cache|batch_set_cache|increment_pipeline"
    r"|rpush|lpop|scan_iter|get_ttl)\("
)

_DECLARES_TARGET: Final = re.compile(r"\b(?:with_service_target|service_target|response_cache_phase)\(")


def _request_path_producers() -> tuple[Path, ...]:
    files: Final = (
        path for directory in _REQUEST_PATH_PRODUCER_DIRS for path in sorted((_REPO / directory).rglob("*.py"))
    )  # comprehension-ok: flatten the producer directories
    return tuple(path for path in files if _CACHE_CALL.search(path.read_text()))


def test_every_request_path_redis_producer_declares_a_key_family() -> None:
    """A proxy hook or routing strategy that reads or writes the cache without a declared
    target renders as a bare ``redis.get`` / ``redis.mget`` flat under the request span, which
    is exactly what the sensitive-data pin read and the rate-limiter MGET did in production."""
    producers: Final = _request_path_producers()
    assert len(producers) >= 20, "the scan no longer finds the known producers; the regex is stale"
    undeclared: Final = tuple(
        path.relative_to(_REPO).as_posix() for path in producers if not _DECLARES_TARGET.search(path.read_text())
    )
    assert undeclared == ()


def test_with_service_target_sets_the_target_for_sync_and_async_calls_and_restores_it() -> None:
    @with_service_target("rate_limits")
    def read() -> str | None:
        return current_service_target()

    @with_service_target("rate_limits")
    async def read_async() -> str | None:
        await asyncio.sleep(0)
        return current_service_target()

    assert read() == "rate_limits"
    assert asyncio.run(read_async()) == "rate_limits"
    assert current_service_target() is None
    with service_target("auth_objects"):
        assert read() == "rate_limits"
        assert current_service_target() == "auth_objects"


def test_with_service_target_keeps_the_wrapped_signature_and_coroutine_ness() -> None:
    import inspect

    @with_service_target("rate_limits")
    async def hook(self: object, data: dict[str, str], call_type: str) -> None:
        return None

    assert inspect.iscoroutinefunction(hook)
    assert tuple(inspect.signature(hook).parameters) == ("self", "data", "call_type")
    assert hook.__name__ == "hook"


def test_service_caller_is_inherited_by_a_task_spawned_inside_it_and_cleared_after() -> None:
    async def spawned() -> str | None:
        return current_service_caller()

    async def main() -> tuple[str | None, str | None]:
        with service_caller("prefetch <- auth"):
            task = asyncio.create_task(spawned())
        return await task, current_service_caller()

    assert asyncio.run(main()) == ("prefetch <- auth", None)


@pytest.mark.parametrize("value", [None, "x"])
def test_service_caller_restores_the_outer_value(value: str | None) -> None:
    with service_caller(value):
        with service_caller("inner"):
            assert current_service_caller() == "inner"
        assert current_service_caller() == value
    assert current_service_caller() is None


class _Suspend:
    def __await__(self) -> Generator[None]:
        yield


def test_a_targeted_coroutine_closed_from_another_context_does_not_raise() -> None:
    @with_service_target("router_usage")
    async def sync_forever() -> None:
        await _Suspend()

    suspended: Final = sync_forever()
    contextvars.copy_context().run(suspended.send, None)
    contextvars.copy_context().run(suspended.close)
    assert current_service_target() is None
