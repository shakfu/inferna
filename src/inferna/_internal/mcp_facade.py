"""MCP-client facade for :class:`inferna.LLM`.

The methods that previously lived directly on ``LLM`` (``add_mcp_server``,
``remove_mcp_server``, ``list_mcp_tools``, ``list_mcp_resources``,
``call_mcp_tool``, ``read_mcp_resource``) all deferred to a single
``McpClient`` from ``inferna.agents.mcp``. Extracting them onto a
dedicated facade leaves the public API on ``LLM`` unchanged (each LLM
method is now a one-line delegate) but isolates the MCP lifecycle
concerns -- lazy client construction, transport-type inference, fail-
fast connect on add, idempotent disconnect on remove -- in one place.

The facade owns ``self._client`` lazily: until the first
``add_mcp_server`` call it is ``None`` and read-only ops (``list_*``)
short-circuit. ``close()`` tears down the client; subsequent ``add``
calls reconstruct one. This matches the contract documented on
``LLM.close()`` (the LLM remains usable after close; new contexts /
clients are made on demand).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..agents.mcp import McpClient, McpResource, McpTool

logger = logging.getLogger(__name__)


class MCPFacade:
    """Lazily holds an :class:`McpClient` and forwards LLM-shaped calls."""

    def __init__(self) -> None:
        self._client: Optional["McpClient"] = None

    def has_client(self) -> bool:
        """True iff at least one ``add_mcp_server`` call has happened."""
        return self._client is not None

    def get_or_create(self) -> "McpClient":
        """Return the client, constructing it on first use."""
        if self._client is None:
            from ..agents.mcp import McpClient

            self._client = McpClient()
        return self._client

    def add_server(
        self,
        name: str,
        *,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        transport: Optional[Any] = None,
        request_timeout: Optional[float] = None,
        shutdown_timeout: Optional[float] = None,
    ) -> None:
        """Attach an MCP server and connect immediately.

        Transport is inferred when ``transport`` is omitted: ``command``
        => stdio, ``url`` => http. Connecting fail-fast (rather than
        deferring to ``connect_all``) is intentional -- callers expect
        ``add_mcp_server`` to surface bad configs / unreachable
        endpoints synchronously.
        """
        from ..agents.mcp import (
            DEFAULT_REQUEST_TIMEOUT,
            DEFAULT_SHUTDOWN_TIMEOUT,
            McpServerConfig,
            McpTransportType,
        )

        if transport is None:
            if command is not None:
                transport = McpTransportType.STDIO
            elif url is not None:
                transport = McpTransportType.HTTP
            else:
                raise ValueError(
                    "add_mcp_server requires either 'command' (stdio) or 'url' (http), or an explicit 'transport'."
                )

        config = McpServerConfig(
            name=name,
            transport=transport,
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            url=url,
            headers=headers,
            request_timeout=request_timeout if request_timeout is not None else DEFAULT_REQUEST_TIMEOUT,
            shutdown_timeout=shutdown_timeout if shutdown_timeout is not None else DEFAULT_SHUTDOWN_TIMEOUT,
        )

        client = self.get_or_create()
        client.add_server(config)
        # Connect this single server now rather than deferring to a global
        # connect_all(): callers expect add_mcp_server() to fail-fast on a
        # bad config or unreachable endpoint.
        client._connect_server(config)
        client._connected = True

    def remove_server(self, name: str) -> None:
        """Disconnect and forget an MCP server by name. No-op if absent."""
        if self._client is None:
            return
        client = self._client
        conn = client._connections.pop(name, None)
        if conn is not None:
            try:
                conn.disconnect()
            except Exception as exc:
                # Disconnect failures shouldn't block forgetting the
                # server; surface via warning so a wedged transport is
                # observable rather than silent.
                logger.warning(
                    "MCP server %r disconnect failed (%s); continuing teardown",
                    name,
                    type(exc).__name__,
                )
        client._servers = [s for s in client._servers if s.name != name]
        # Drop tools and resources owned by this server.
        client._tools = {k: v for k, v in client._tools.items() if v.server_name != name}
        client._resources = {k: v for k, v in client._resources.items() if v.server_name != name}
        if not client._connections:
            client._connected = False

    def list_tools(self) -> List["McpTool"]:
        if self._client is None:
            return []
        return list(self._client.get_tools())

    def list_resources(self) -> List["McpResource"]:
        if self._client is None:
            return []
        return list(self._client.get_resources())

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("No MCP servers attached. Call add_mcp_server() first.")
        return self._client.call_tool(name, arguments)

    def read_resource(self, uri: str) -> str:
        if self._client is None:
            raise RuntimeError("No MCP servers attached. Call add_mcp_server() first.")
        return self._client.read_resource(uri)

    def get_tools_for_agent(self) -> List[Any]:
        """Return MCP tools wrapped for ReActAgent. Empty if no client."""
        if self._client is None:
            return []
        return list(self._client.get_tools_for_agent())

    def close(self) -> None:
        """Disconnect all servers and drop the client.

        Best-effort: transports already log their own errors and we don't
        want a flaky remote server to block local resource cleanup.
        """
        if self._client is None:
            return
        try:
            self._client.disconnect_all()
        except Exception as exc:
            logger.warning(
                "MCP disconnect_all failed (%s); dropping client anyway",
                type(exc).__name__,
            )
        self._client = None
