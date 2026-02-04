"""Example usage of LangChain instrumentation.

This example demonstrates how to use Prela's auto-instrumentation
for LangChain applications.

NOTE: This example requires LangChain to be installed:
    pip install langchain-core langchain-openai

For actual API calls, you'll also need:
    export OPENAI_API_KEY=your_key_here
"""

from __future__ import annotations

import os

# Example 1: Basic Auto-Instrumentation
def example_basic_instrumentation():
    """Most basic usage - auto-instrument all LangChain operations."""
    import prela
    from prela.exporters.console import ConsoleExporter

    # Initialize Prela with console output
    prela.init(
        service_name="langchain-basic-example",
        exporter=ConsoleExporter(),
    )

    # Now all LangChain operations are automatically traced!
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema.output_parser import StrOutputParser

        # Create a simple chain
        prompt = ChatPromptTemplate.from_template(
            "Tell me a short joke about {topic}"
        )
        llm = ChatOpenAI(temperature=0.9)
        output_parser = StrOutputParser()

        chain = prompt | llm | output_parser

        # This execution is automatically traced
        result = chain.invoke({"topic": "programming"})
        print(f"\nJoke: {result}\n")

    except ImportError:
        print("Please install langchain-openai: pip install langchain-openai")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable")


# Example 2: Manual Instrumentation with Custom Tracer
def example_manual_instrumentation():
    """Manual instrumentation for more control."""
    from prela.core.tracer import Tracer
    from prela.instrumentation.langchain import LangChainInstrumentor
    from prela.exporters.file import FileExporter

    # Create custom tracer with file export
    tracer = Tracer(
        service_name="langchain-manual-example",
        exporter=FileExporter(file_path="langchain_traces.jsonl"),
    )

    # Manually instrument LangChain
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer)

    print(f"LangChain instrumented: {instrumentor.is_instrumented}")

    try:
        from langchain_openai import OpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        # Create chain
        llm = OpenAI(temperature=0.7)
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        # Execute chain - automatically traced
        result = chain.run("eco-friendly water bottles")
        print(f"\nCompany name suggestion: {result}\n")

        # Uninstrument when done
        instrumentor.uninstrument()
        print(f"LangChain instrumented: {instrumentor.is_instrumented}")

    except ImportError:
        print("Please install langchain-openai: pip install langchain-openai")
    except Exception as e:
        print(f"Error: {e}")


# Example 3: Tracing Agent Operations
def example_agent_tracing():
    """Trace agent execution with tools."""
    import prela

    prela.init(
        service_name="langchain-agent-example",
        exporter="console",
    )

    try:
        from langchain_openai import OpenAI
        from langchain.agents import AgentType, initialize_agent, load_tools

        # Initialize agent with tools
        llm = OpenAI(temperature=0)
        tools = load_tools(["llm-math"], llm=llm)
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

        # Execute agent - all steps traced
        result = agent.run(
            "What is 15 multiplied by 7, then add 30, then divide by 5?"
        )
        print(f"\nAgent result: {result}\n")

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install langchain-openai langchain-community")
    except Exception as e:
        print(f"Error: {e}")


# Example 4: Concurrent Chain Execution
def example_concurrent_chains():
    """Trace multiple concurrent chain executions."""
    import asyncio
    import prela

    prela.init(
        service_name="langchain-concurrent-example",
        exporter="console",
    )

    async def run_multiple_chains():
        try:
            from langchain_openai import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema.output_parser import StrOutputParser

            llm = ChatOpenAI(temperature=0.9)
            output_parser = StrOutputParser()

            # Create multiple chains
            topics = ["cats", "space", "cooking"]
            chains = []

            for topic in topics:
                prompt = ChatPromptTemplate.from_template(
                    f"Tell me an interesting fact about {topic}"
                )
                chain = prompt | llm | output_parser
                chains.append(chain.ainvoke({}))

            # Execute concurrently - each gets its own trace
            results = await asyncio.gather(*chains)

            for topic, result in zip(topics, results):
                print(f"\n{topic.title()}: {result}")

        except ImportError:
            print("Please install langchain-openai: pip install langchain-openai")
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(run_multiple_chains())


# Example 5: Error Handling
def example_error_handling():
    """Demonstrate error tracking in traces."""
    import prela

    prela.init(
        service_name="langchain-error-example",
        exporter="console",
    )

    try:
        from langchain_openai import OpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        # Create chain that will fail (invalid API key)
        llm = OpenAI(
            temperature=0.7,
            openai_api_key="invalid_key_for_testing",
        )
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Write a poem about {topic}",
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        try:
            # This will fail, but the error is captured in the trace
            result = chain.run("testing")
        except Exception as e:
            print(f"\nExpected error occurred: {type(e).__name__}")
            print("Check the trace output - error details are captured!\n")

    except ImportError:
        print("Please install langchain-openai: pip install langchain-openai")


# Example 6: Using Callback Handler Directly
def example_direct_callback_usage():
    """Use the callback handler directly on specific operations."""
    from prela.core.tracer import Tracer
    from prela.instrumentation.langchain import LangChainInstrumentor
    from prela.exporters.console import ConsoleExporter

    tracer = Tracer(
        service_name="langchain-callback-example",
        exporter=ConsoleExporter(),
    )

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer)

    # Get the callback handler
    handler = instrumentor.get_callback()

    try:
        from langchain_openai import OpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        llm = OpenAI(temperature=0.7)
        prompt = PromptTemplate(
            input_variables=["color"],
            template="Suggest a creative name for a {color} car",
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        # Explicitly pass callback to this specific operation
        result = chain.run("blue", callbacks=[handler] if handler else [])
        print(f"\nCar name: {result}\n")

    except ImportError:
        print("Please install langchain-openai: pip install langchain-openai")
    except Exception as e:
        print(f"Error: {e}")


# Main
if __name__ == "__main__":
    print("=" * 80)
    print("LangChain Instrumentation Examples")
    print("=" * 80)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set")
        print("Most examples will fail without a valid API key.")
        print("Set it with: export OPENAI_API_KEY=your_key_here\n")

    # Run examples (comment out ones you don't want to run)
    print("\n--- Example 1: Basic Auto-Instrumentation ---")
    example_basic_instrumentation()

    print("\n--- Example 2: Manual Instrumentation ---")
    example_manual_instrumentation()

    # Uncomment to run agent example (requires serpapi or other tools)
    # print("\n--- Example 3: Agent Tracing ---")
    # example_agent_tracing()

    # Uncomment to run concurrent example
    # print("\n--- Example 4: Concurrent Chains ---")
    # example_concurrent_chains()

    print("\n--- Example 5: Error Handling ---")
    example_error_handling()

    print("\n--- Example 6: Direct Callback Usage ---")
    example_direct_callback_usage()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
