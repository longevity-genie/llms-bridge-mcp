"""Tests for LLMs Bridge MCP Server."""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pytest
from dotenv import load_dotenv
from llms_bridge_mcp.server import LLMBridgeMCP


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file before running tests."""
    load_dotenv()


def test_server_initialization():
    """Test server initializes correctly."""
    server = LLMBridgeMCP()
    assert server is not None
    assert hasattr(server, "prefix")
    assert server.prefix == "llms_bridge_"


@pytest.mark.asyncio
async def test_server_has_tools():
    """Test server has registered tools dynamically based on API keys."""
    server = LLMBridgeMCP()
    tools = await server.list_tools()
    assert len(tools) > 0

    tool_names = [tool.name for tool in tools]

    # Check which tools are registered based on available API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    if openai_key:
        assert "llms_bridge_call_chatgpt" in tool_names
    if gemini_key:
        assert "llms_bridge_call_gemini" in tool_names
    if perplexity_key:
        assert "llms_bridge_call_perplexity" in tool_names

    # If no keys are present, should have misconfiguration tool
    if not openai_key and not gemini_key and not perplexity_key:
        assert "llms_bridge_server_misconfiguration" in tool_names


@pytest.mark.asyncio
async def test_server_misconfiguration():
    """Test server misconfiguration handler when no API keys are present."""
    # Temporarily clear environment variables
    original_openai = os.environ.pop("OPENAI_API_KEY", None)
    original_gemini = os.environ.pop("GEMINI_API_KEY", None)
    original_perplexity = os.environ.pop("PERPLEXITY_API_KEY", None)

    try:
        # Create server without any API keys
        server = LLMBridgeMCP()

        # Should have the misconfiguration tool
        tools = await server.list_tools()
        tool_names = [tool.name for tool in tools]
        assert "llms_bridge_server_misconfiguration" in tool_names

        # Call the misconfiguration tool
        result = await server.server_misconfiguration()
        assert not result.success
        assert "without any API keys" in result.data["configuration_instructions"]
        assert "OPENAI_API_KEY" in result.data["configuration_instructions"]
        assert "GEMINI_API_KEY" in result.data["configuration_instructions"]
        assert "PERPLEXITY_API_KEY" in result.data["configuration_instructions"]
    finally:
        # Restore environment variables
        if original_openai:
            os.environ["OPENAI_API_KEY"] = original_openai
        if original_gemini:
            os.environ["GEMINI_API_KEY"] = original_gemini
        if original_perplexity:
            os.environ["PERPLEXITY_API_KEY"] = original_perplexity


@pytest.mark.asyncio
async def test_battle_real_llm_calls():
    """
    Battle test: Call real LLMs with actual queries and save responses.

    This test requires API keys to be set in environment.
    Supported keys: OPENAI_API_KEY, GEMINI_API_KEY, PERPLEXITY_API_KEY

    It asks: What date is today in UTC+00 in ISO format? What is the weather in Kotor?

    Results are saved to tests/battle_test_results.json
    """
    # Skip if API keys are not set
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    if not openai_key and not gemini_key and not perplexity_key:
        pytest.skip(
            "No API keys configured. Set OPENAI_API_KEY, GEMINI_API_KEY, or PERPLEXITY_API_KEY to run battle test"
        )

    server = LLMBridgeMCP()

    # Queries to test
    queries = [
        "What date is today in UTC+00 in ISO format? What is the weather in Kotor?"
    ]

    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "queries": queries,
        "responses": {},
    }

    # Test ChatGPT if key is available - use default model from server
    if openai_key:
        print("\n=== Testing ChatGPT ===")
        results["responses"]["chatgpt"] = {}

        for query in queries:
            print(f"\nQuery: {query}")
            result = await server.call_chatgpt(prompt=query)

            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            if result.success and "response" in result.data:
                print(f"Response: {result.data['response']}")
                print(f"Model: {result.data.get('model', 'unknown')}")
                print(f"Tokens: {result.data.get('usage', {})}")
            else:
                print(f"Error data: {result.data}")

            results["responses"]["chatgpt"][query] = {
                "response": result.data.get("response") if result.success else None,
                "error": result.data.get("error") if not result.success else None,
                "model": result.data.get("model"),
                "usage": result.data.get("usage"),
                "success": result.success,
                "message": result.message,
            }

    # Test Gemini if key is available - use default model from server
    if gemini_key:
        print("\n=== Testing Gemini ===")
        results["responses"]["gemini"] = {}

        for query in queries:
            print(f"\nQuery: {query}")
            result = await server.call_gemini(prompt=query)

            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            if result.success and "response" in result.data:
                print(f"Response: {result.data['response']}")
                print(f"Model: {result.data.get('model', 'unknown')}")
                print(f"Tokens: {result.data.get('usage', {})}")
            else:
                print(f"Error data: {result.data}")

            results["responses"]["gemini"][query] = {
                "response": result.data.get("response") if result.success else None,
                "error": result.data.get("error") if not result.success else None,
                "model": result.data.get("model"),
                "usage": result.data.get("usage"),
                "success": result.success,
                "message": result.message,
            }

    # Test Perplexity if key is available - use default model from server
    if perplexity_key:
        print("\n=== Testing Perplexity ===")
        results["responses"]["perplexity"] = {}

        for query in queries:
            print(f"\nQuery: {query}")
            result = await server.call_perplexity(prompt=query)

            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            if result.success and "response" in result.data:
                print(f"Response: {result.data['response']}")
                print(f"Model: {result.data.get('model', 'unknown')}")
                print(f"Tokens: {result.data.get('usage', {})}")
            else:
                print(f"Error data: {result.data}")

            results["responses"]["perplexity"][query] = {
                "response": result.data.get("response") if result.success else None,
                "error": result.data.get("error") if not result.success else None,
                "model": result.data.get("model"),
                "usage": result.data.get("usage"),
                "success": result.success,
                "message": result.message,
            }

    # Save results to file
    output_file = Path(__file__).parent / "battle_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\n✅ Battle test results saved to: {output_file}")
    print(f"Total queries tested: {len(queries)}")
    print(f"LLMs tested: {', '.join(results['responses'].keys())}")

    # Get current UTC date in ISO format for assertion
    current_utc_date = datetime.now(timezone.utc).date().isoformat()
    print(f"\nCurrent UTC date: {current_utc_date}")

    # Basic assertions
    assert isinstance(results, dict)
    for llm_name, llm_results in results["responses"].items():
        for query, query_result in llm_results.items():
            if query_result["success"]:
                assert isinstance(query_result["response"], str)

                # Check if the date query response contains current UTC date
                if "date" in query.lower() and "iso format" in query.lower():
                    response_text = query_result["response"]
                    assert current_utc_date in response_text, (
                        f"{llm_name} response should contain current UTC date {current_utc_date}, but got: {response_text[:200]}"
                    )
                    print(
                        f"✅ {llm_name}: Found current UTC date {current_utc_date} in response"
                    )
