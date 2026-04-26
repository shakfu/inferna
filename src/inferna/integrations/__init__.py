"""
Integration helpers for popular Python frameworks.

This module provides adapters and utilities to integrate inferna with
popular frameworks like LangChain, OpenAI API, and others.
"""

from .langchain import InfernaLLM
from .openai_compat import OpenAICompatibleClient as OpenAIClient

# Agent integrations (optional - require agent module)
try:
    from .langchain_agents import (
        inferna_tool_to_langchain,
        langchain_tool_to_inferna,
        create_langchain_agent_executor,
        InfernaAgentLangChainAdapter,
        create_inferna_react_agent,
        create_inferna_constrained_agent,
    )
    from .openai_agents import (
        OpenAIFunctionCallingClient,
        create_openai_function_calling_client,
        inferna_tool_to_openai_function,
        inferna_tools_to_openai_tools,
    )

    AGENT_INTEGRATIONS_AVAILABLE = True
except ImportError:
    AGENT_INTEGRATIONS_AVAILABLE = False

__all__ = [
    "InfernaLLM",
    "OpenAIClient",
]

if AGENT_INTEGRATIONS_AVAILABLE:
    __all__.extend(
        [
            # LangChain agents
            "inferna_tool_to_langchain",
            "langchain_tool_to_inferna",
            "create_langchain_agent_executor",
            "InfernaAgentLangChainAdapter",
            "create_inferna_react_agent",
            "create_inferna_constrained_agent",
            # OpenAI function calling
            "OpenAIFunctionCallingClient",
            "create_openai_function_calling_client",
            "inferna_tool_to_openai_function",
            "inferna_tools_to_openai_tools",
        ]
    )
