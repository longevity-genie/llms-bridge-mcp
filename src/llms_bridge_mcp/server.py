#!/usr/bin/env python3
"""LLMs Bridge MCP Server - Interface for bridging LLM calls via Model Context Protocol."""

import os
import sys
import signal
from openai.types.shared import reasoning_effort
import typer

from typing import Any, Optional, Dict, List
from importlib.metadata import version, PackageNotFoundError
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from litellm import acompletion, drop_params


# Configuration
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "3011"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")
DEFAULT_TIMEOUT = int(os.getenv("MCP_TIMEOUT", "120"))

# Get package version
try:
    __version__ = version("llms-bridge-mcp")
except PackageNotFoundError:
    __version__ = "0.1.0"


class LLMBridgeResult(BaseModel):
    """Result from an LLMs Bridge operation."""
    data: Any = Field(description="Response data from the operation")
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Operation description")


class LLMBridgeMCP(FastMCP):
    """LLMs Bridge MCP Server with LLM integration tools."""
    
    def __init__(
        self, 
        name: str = f"LLMs Bridge MCP Server v{__version__}",
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        perplexity_api_key: Optional[str] = None,
        prefix: str = "llms_bridge_",
        **kwargs
    ):
        """Initialize the LLMs Bridge MCP server with FastMCP functionality."""
        super().__init__(name=name, **kwargs)
        
        # Get API credentials from environment if not provided
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.perplexity_api_key = perplexity_api_key or os.getenv("PERPLEXITY_API_KEY", "")
        self.prefix = prefix
        
        # Set environment variables for litellm
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key
        if self.perplexity_api_key:
            os.environ["PERPLEXITY_API_KEY"] = self.perplexity_api_key
        
        # Register tools and resources
        self._register_tools()
        self._register_resources()
    
    async def _call_llm(
        self,
        query: str,
        model: str,
        **kwargs
    ) -> LLMBridgeResult:
        """
        Internal method to call any LLM via litellm.
        
        Args:
            query: The user message/prompt
            model: The model identifier (e.g., "gpt/5.2", "gemini/gemini-3-pro-preview")
            **kwargs: Additional parameters (temperature, max_tokens, system_message, reasoning_effort, etc.)
            
        Returns:
            LLMBridgeResult containing the LLM response
        """
        try:
            messages: List[Dict[str, str]] = []
            stream = kwargs.pop("stream", False) and False #force non-streaming by default
            system_message = kwargs.pop("system_message", None)
            reasoning_effort = kwargs.pop("reasoning_effort", "low")
            if reasoning_effort not in ["none", "minimal", "low", "medium", "high", "xhigh"]:
                reasoning_effort = "low"
            web_search_context_size = kwargs.pop("web_search_context_size", "low")
            if web_search_context_size not in ["low", "medium", "high"]:
                web_search_context_size = "medium"

            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": query})
            
            # Build completion kwargs
            completion_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "web_search_options": {
                    "search_context_size": web_search_context_size  # Options: "low", "medium" (default), "high"
                },
            }
            
            if not "gemini" in model:
                completion_kwargs.pop("web_search_options") # not supported by non-Gemini models

            # Add additional kwargs (temperature, max_tokens, etc.)
            completion_kwargs.update(kwargs)
            
            response = await acompletion(
                **completion_kwargs,
                stream=stream,
                reasoning_effort=reasoning_effort,
    
                timeout=DEFAULT_TIMEOUT,
                drop_params=True #drop all params that are not supported by the model
            )
            
            return LLMBridgeResult(
                data={
                    "response": response.choices[0].message.content,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                },
                success=True,
                message=f"Successfully called LLM ({model})"
            )
        except Exception as e:
            return LLMBridgeResult(
                data={"error": str(e)},
                success=False,
                message=f"Failed to call LLM: {str(e)}"
            )
    
    def _register_tools(self):
        """Register LLMs Bridge tools dynamically based on available API keys."""
        has_any_key = False
        
        if self.openai_api_key:
            self.tool(
                name=f"{self.prefix}call_chatgpt",
                description="Call OpenAI ChatGPT API with a prompt and optional conversation history"
            )(self.call_chatgpt)
            has_any_key = True
        
        if self.gemini_api_key:
            self.tool(
                name=f"{self.prefix}call_gemini",
                description="Call Google Gemini API with a prompt and optional conversation history"
            )(self.call_gemini)
            has_any_key = True
        
        if self.perplexity_api_key:
            self.tool(
                name=f"{self.prefix}call_perplexity",
                description="Call Perplexity Sonar API with a prompt for web-search enhanced responses"
            )(self.call_perplexity)
            has_any_key = True
        
        # If no API keys are present, register a misconfiguration tool
        if not has_any_key:
            self.tool(
                name=f"{self.prefix}server_misconfiguration",
                description="Information about server configuration status"
            )(self.server_misconfiguration)
    
    def _register_resources(self):
        """Register LLMs Bridge resources."""
        
        @self.resource(f"resource://{self.prefix}info")
        def get_info() -> str:
            """
            Get information about the LLMs Bridge MCP server.
            
            Returns:
                Server information and usage guidelines
            """
            return f"""
            # LLMs Bridge MCP Server v{__version__}
            
            ## Overview
            This is an MCP server for bridging LLM calls via the Model Context Protocol.
            
            ## Available Tools
            
            ### llms_bridge_call_chatgpt
            - **Description**: Call OpenAI ChatGPT API with a prompt
            - **Parameters**: 
              - prompt (required): The user message/prompt
              - model (optional): OpenAI model (default: gpt-5.1)
              - temperature (optional): Sampling temperature 0-2 (default: 0.0)
              - max_tokens (optional): Maximum response tokens
              - system_message (optional): System message for context
            - **Example**: `call_chatgpt(prompt="Explain quantum computing", model="gpt-5.1")`
            - **Status**: {"Available" if self.openai_api_key else "Not available - OPENAI_API_KEY not set"}
            
            ### llms_bridge_call_gemini
            - **Description**: Call Google Gemini API with a prompt
            - **Parameters**: 
              - prompt (required): The user message/prompt
              - model (optional): Gemini model (default: gemini/gemini-3-pro-preview)
              - temperature (optional): Sampling temperature 0-2 (default: 0.0)
              - max_tokens (optional): Maximum response tokens
              - system_message (optional): System message for context
            - **Example**: `call_gemini(prompt="What is machine learning?", model="gemini/gemini-3-pro-preview")`
            - **Status**: {"Available" if self.gemini_api_key else "Not available - GEMINI_API_KEY not set"}
            
            ### llms_bridge_call_perplexity
            - **Description**: Call Perplexity Sonar API with web-search enhanced responses
            - **Parameters**: 
              - prompt (required): The user message/prompt
              - model (optional): Perplexity model (default: perplexity/sonar)
              - temperature (optional): Sampling temperature 0-2 (default: 0.0)
              - max_tokens (optional): Maximum response tokens
              - system_message (optional): System message for context
            - **Example**: `call_perplexity(prompt="What are the latest AI developments?", model="perplexity/sonar")`
            - **Status**: {"Available" if self.perplexity_api_key else "Not available - PERPLEXITY_API_KEY not set"}
            
            ## Configuration
            - **OpenAI API Key**: {"Set" if self.openai_api_key else "Not set"} (via OPENAI_API_KEY)
            - **Gemini API Key**: {"Set" if self.gemini_api_key else "Not set"} (via GEMINI_API_KEY)
            - **Perplexity API Key**: {"Set" if self.perplexity_api_key else "Not set"} (via PERPLEXITY_API_KEY)
            - **Version**: {__version__}
            """
    
    async def call_chatgpt(
        self,
        prompt: str,
        model: str = "gpt-5.1",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> LLMBridgeResult:
        """
        Call OpenAI ChatGPT API using litellm.
        
        Args:
            prompt: The user message/prompt to send to ChatGPT
            model: The OpenAI model to use (default: gpt-5.1)
            temperature: Sampling temperature between 0 and 2 (default: 0.7)
            max_tokens: Maximum tokens in the response (optional)
            system_message: Optional system message to set context
            
        Returns:
            LLMBridgeResult containing the ChatGPT response
        """
        if not self.openai_api_key:
            return LLMBridgeResult(
                data={},
                success=False,
                message="OPENAI_API_KEY is not configured"
            )
        
        kwargs: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if system_message:
            kwargs["system_message"] = system_message
        
        return await self._call_llm(query=prompt, model=model, **kwargs)
    
    async def call_gemini(
        self,
        prompt: str,
        model: str = "gemini/gemini-3-pro-preview",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> LLMBridgeResult:
        """
        Call Google Gemini API using litellm.
        
        Args:
            prompt: The user message/prompt to send to Gemini
            model: The Gemini model to use (default: gemini/gemini-3-pro-preview)
            temperature: Sampling temperature between 0 and 2 (default: 0.7)
            max_tokens: Maximum tokens in the response (optional)
            system_message: Optional system message to set context
            
        Returns:
            LLMBridgeResult containing the Gemini response
        """
        if not self.gemini_api_key:
            return LLMBridgeResult(
                data={},
                success=False,
                message="GEMINI_API_KEY is not configured"
            )
        
        kwargs: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if system_message:
            kwargs["system_message"] = system_message
        
        return await self._call_llm(query=prompt, model=model, **kwargs)
    
    async def call_perplexity(
        self,
        prompt: str,
        model: str = "perplexity/sonar",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> LLMBridgeResult:
        """
        Call Perplexity Sonar API using litellm.
        
        Args:
            prompt: The user message/prompt to send to Perplexity
            model: The Perplexity model to use (default: perplexity/sonar)
            temperature: Sampling temperature between 0 and 2 (default: 0.0)
            max_tokens: Maximum tokens in the response (optional)
            system_message: Optional system message to set context
            
        Returns:
            LLMBridgeResult containing the Perplexity response
        """
        if not self.perplexity_api_key:
            return LLMBridgeResult(
                data={},
                success=False,
                message="PERPLEXITY_API_KEY is not configured"
            )
        
        kwargs: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if system_message:
            kwargs["system_message"] = system_message
        
        return await self._call_llm(query=prompt, model=model, **kwargs)
    
    async def server_misconfiguration(self) -> LLMBridgeResult:
        """
        Return information about server misconfiguration when no API keys are present.
        
        Returns:
            LLMBridgeResult with configuration instructions
        """
        supported_keys = [
            "OPENAI_API_KEY - for ChatGPT models",
            "GEMINI_API_KEY - for Google Gemini models",
            "PERPLEXITY_API_KEY - for Perplexity Sonar models"
        ]
        
        message = (
            "⚠️ LLMs Bridge MCP Server was started without any API keys configured.\n\n"
            "Supported API keys:\n" + "\n".join(f"  • {key}" for key in supported_keys) + "\n\n"
            "To configure the server:\n"
            "1. Set one or more of the supported API keys as environment variables\n"
            "2. Restart the MCP server\n\n"
            "Example:\n"
            "  export OPENAI_API_KEY=your_key_here\n"
            "  export GEMINI_API_KEY=your_key_here\n"
            "  export PERPLEXITY_API_KEY=your_key_here\n"
        )
        
        return LLMBridgeResult(
            data={"configuration_instructions": message},
            success=False,
            message="Server is misconfigured - no API keys present"
        )


def get_mcp_server():
    """Get or create the MCP server instance."""
    return LLMBridgeMCP()


class GracefulShutdownHandler:
    """Handle graceful shutdown on SIGINT and SIGTERM."""
    
    def __init__(self):
        self.shutdown_requested = False
        self.original_sigint_handler = None
        self.original_sigterm_handler = None
    
    def register_handlers(self):
        """Register signal handlers for graceful shutdown."""
        self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self.original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
    
    def restore_handlers(self):
        """Restore original signal handlers."""
        if self.original_sigint_handler:
            signal.signal(signal.SIGINT, self.original_sigint_handler)
        if self.original_sigterm_handler:
            signal.signal(signal.SIGTERM, self.original_sigterm_handler)
    
    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals gracefully."""
        if self.shutdown_requested:
            # Force exit on second signal to avoid hanging
            os._exit(1)
        
        self.shutdown_requested = True
        raise KeyboardInterrupt()


def run_with_graceful_shutdown(server: LLMBridgeMCP, **kwargs):
    """Run the server with graceful shutdown handling.
    
    Args:
        server: The LLMBridgeMCP server instance
        **kwargs: Additional arguments to pass to server.run()
    """
    shutdown_handler = GracefulShutdownHandler()
    exit_code = 0
    should_force_exit = True
    
    try:
        shutdown_handler.register_handlers()
        server.run(**kwargs)
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        pass
    except EOFError:
        # Handle EOF gracefully (e.g., when stdin is closed)
        pass
    except Exception as e:
        error_msg = str(e)
        # Ignore JSON parsing errors during shutdown
        if "EOF while parsing" in error_msg or "JSONRPCMessage" in error_msg:
            pass
        else:
            print(f"Server error: {error_msg}", file=sys.stderr)
            exit_code = 1
            should_force_exit = False
            raise
    finally:
        shutdown_handler.restore_handlers()
        # Flush output streams
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except:
            pass
        
        # Force exit to avoid daemon thread cleanup issues on clean shutdown
        if should_force_exit:
            os._exit(exit_code)


class SmitheryConfigSchema(BaseModel):
    """Configuration schema for Smithery."""
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key (optional)")
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key (optional)")
    perplexity_api_key: Optional[str] = Field(default=None, description="Perplexity API key (optional)")


# CLI application using typer
app = typer.Typer()


@app.command()
def main(
    host: str = typer.Option(DEFAULT_HOST, help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, help="Port to bind to"),
    transport: str = typer.Option(DEFAULT_TRANSPORT, help="Transport type")
):
    """Run the LLMs Bridge MCP server."""
    run_with_graceful_shutdown(get_mcp_server(), transport=transport, host=host, port=port)


@app.command()
def stdio():
    """Run the LLMs Bridge MCP server with stdio transport."""
    run_with_graceful_shutdown(get_mcp_server(), transport="stdio")


@app.command()
def http(
    host: str = typer.Option(DEFAULT_HOST, help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, help="Port to bind to")
):
    """Run the LLMs Bridge MCP server with HTTP transport."""
    run_with_graceful_shutdown(get_mcp_server(), transport="http", host=host, port=port)


@app.command()
def sse(
    host: str = typer.Option(DEFAULT_HOST, help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, help="Port to bind to")
):
    """Run the LLMs Bridge MCP server with SSE transport."""
    run_with_graceful_shutdown(get_mcp_server(), transport="sse", host=host, port=port)


def cli_app():
    """Entry point for the CLI application."""
    app()


def cli_app_stdio():
    """Entry point for stdio mode."""
    stdio()


def cli_app_sse():
    """Entry point for SSE mode."""
    sse()


def cli_app_http():
    """Entry point for HTTP mode."""
    http()


if __name__ == "__main__":
    app()
