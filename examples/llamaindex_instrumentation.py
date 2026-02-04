"""Example demonstrating LlamaIndex instrumentation with Prela.

This script shows how to automatically trace LlamaIndex operations including:
- Query engine queries
- Vector retrieval with similarity scores
- LLM calls through LlamaIndex
- Embedding generation
- Response synthesis

Requirements:
    pip install prela llama-index llama-index-llms-openai llama-index-embeddings-openai

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY=your_api_key_here

    # Run the examples
    python examples/llamaindex_instrumentation.py
"""

from __future__ import annotations


def example_basic_query():
    """Example 1: Basic query engine with auto-tracing."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Query Engine")
    print("=" * 60)

    import prela
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    # Initialize Prela with auto-instrumentation
    prela.init(
        service_name="llamaindex-demo",
        exporter="console",
        verbosity="normal",
    )

    # Configure LlamaIndex (these calls will be auto-traced)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

    # Create some sample documents
    from llama_index.core import Document

    documents = [
        Document(
            text="Python is a high-level programming language known for its simplicity.",
            metadata={"source": "programming_guide.txt"},
        ),
        Document(
            text="Machine learning involves training algorithms on data to make predictions.",
            metadata={"source": "ml_intro.txt"},
        ),
        Document(
            text="LlamaIndex is a framework for building LLM applications with structured data.",
            metadata={"source": "llamaindex_docs.txt"},
        ),
    ]

    # Build index (embedding calls will be traced)
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # Query the index (retrieval + LLM synthesis traced)
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query("What is LlamaIndex?")

    print(f"\nQuery: What is LlamaIndex?")
    print(f"Response: {response}")

    # The trace shows:
    # - Query event (AGENT type)
    # - Embedding generation for query (EMBEDDING type)
    # - Retrieval with node scores (RETRIEVAL type)
    # - LLM synthesis call (LLM type)
    # - Synthesis completion (AGENT type)


def example_custom_retrieval():
    """Example 2: Custom retrieval with similarity scores."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Retrieval")
    print("=" * 60)

    import prela
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding

    prela.init(service_name="llamaindex-retrieval", exporter="console")

    # Create documents
    documents = [
        Document(text="The sky is blue because of Rayleigh scattering."),
        Document(text="Photosynthesis converts light energy into chemical energy."),
        Document(text="Water boils at 100 degrees Celsius at sea level."),
    ]

    # Build index
    embed_model = OpenAIEmbedding()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # Configure retriever with custom top_k
    retriever = index.as_retriever(similarity_top_k=2)

    # Retrieve nodes (shows similarity scores in trace)
    nodes = retriever.retrieve("Why is the sky blue?")

    print(f"\nRetrieved {len(nodes)} nodes:")
    for i, node in enumerate(nodes):
        print(f"{i+1}. Score: {node.score:.3f} - {node.node.text[:50]}...")

    # Trace shows:
    # - RETRIEVE event with query and similarity_top_k
    # - Node scores and text snippets
    # - File metadata if available


def example_chat_engine():
    """Example 3: Chat engine with conversation history."""
    print("\n" + "=" * 60)
    print("Example 3: Chat Engine")
    print("=" * 60)

    import prela
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.llms.openai import OpenAI

    prela.init(service_name="llamaindex-chat", exporter="console", verbosity="verbose")

    # Create knowledge base
    documents = [
        Document(text="Paris is the capital of France."),
        Document(text="The Eiffel Tower is located in Paris."),
        Document(text="France is known for wine, cheese, and baguettes."),
    ]

    index = VectorStoreIndex.from_documents(documents)

    # Create chat engine
    llm = OpenAI(model="gpt-3.5-turbo")
    chat_engine = index.as_chat_engine(llm=llm, chat_mode="context")

    # Chat with context
    response1 = chat_engine.chat("What is the capital of France?")
    print(f"\nUser: What is the capital of France?")
    print(f"Bot: {response1}")

    response2 = chat_engine.chat("What famous landmark is there?")
    print(f"\nUser: What famous landmark is there?")
    print(f"Bot: {response2}")

    # Each chat turn creates a trace with:
    # - Query understanding
    # - Context retrieval
    # - LLM call with chat history
    # - Response synthesis


def example_sub_question_query():
    """Example 4: Sub-question query engine (complex queries)."""
    print("\n" + "=" * 60)
    print("Example 4: Sub-Question Query Engine")
    print("=" * 60)

    import prela
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.query_engine import SubQuestionQueryEngine
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.llms.openai import OpenAI

    prela.init(
        service_name="llamaindex-subquestions", exporter="console", verbosity="verbose"
    )

    # Create separate indexes for different topics
    python_docs = [
        Document(text="Python was created by Guido van Rossum in 1991."),
        Document(text="Python is known for its simple, readable syntax."),
    ]

    ml_docs = [
        Document(text="Machine learning trains models on data."),
        Document(text="Common ML algorithms include linear regression and neural networks."),
    ]

    python_index = VectorStoreIndex.from_documents(python_docs)
    ml_index = VectorStoreIndex.from_documents(ml_docs)

    # Create tools for each index
    python_tool = QueryEngineTool(
        query_engine=python_index.as_query_engine(),
        metadata=ToolMetadata(
            name="python_docs",
            description="Information about Python programming language",
        ),
    )

    ml_tool = QueryEngineTool(
        query_engine=ml_index.as_query_engine(),
        metadata=ToolMetadata(
            name="ml_docs", description="Information about machine learning"
        ),
    )

    # Create sub-question engine
    llm = OpenAI(model="gpt-3.5-turbo")
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[python_tool, ml_tool], llm=llm
    )

    # Complex query requiring multiple sub-questions
    response = query_engine.query(
        "What is Python and how is it used in machine learning?"
    )

    print(f"\nComplex Query: What is Python and how is it used in machine learning?")
    print(f"Response: {response}")

    # Trace shows:
    # - SUB_QUESTION events for query decomposition
    # - Multiple QUERY events (one per sub-question)
    # - RETRIEVE events for each sub-index
    # - LLM calls for each sub-answer
    # - Final SYNTHESIZE event combining results


def example_with_metadata():
    """Example 5: Document retrieval with metadata filtering."""
    print("\n" + "=" * 60)
    print("Example 5: Metadata Filtering")
    print("=" * 60)

    import prela
    from llama_index.core import Document, VectorStoreIndex

    prela.init(service_name="llamaindex-metadata", exporter="console")

    # Create documents with rich metadata
    documents = [
        Document(
            text="Quantum computing uses qubits instead of bits.",
            metadata={
                "file_name": "quantum_intro.txt",
                "page_label": "1",
                "category": "physics",
            },
        ),
        Document(
            text="Blockchain is a distributed ledger technology.",
            metadata={
                "file_name": "blockchain_basics.txt",
                "page_label": "1",
                "category": "technology",
            },
        ),
        Document(
            text="Photosynthesis converts sunlight into chemical energy.",
            metadata={
                "file_name": "biology_101.txt",
                "page_label": "42",
                "category": "biology",
            },
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=2)

    # Retrieve with metadata
    nodes = retriever.retrieve("Tell me about quantum computing")

    print(f"\nRetrieved {len(nodes)} nodes with metadata:")
    for node in nodes:
        metadata = node.node.metadata
        print(f"  - {metadata.get('file_name')} (page {metadata.get('page_label')})")
        print(f"    Category: {metadata.get('category')}")
        print(f"    Score: {node.score:.3f}")

    # Trace shows:
    # - retrieval.node.0.file_name
    # - retrieval.node.0.page_label
    # - retrieval.node.0.score
    # - etc.


def example_error_handling():
    """Example 6: Error handling in LlamaIndex operations."""
    print("\n" + "=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)

    import prela
    from llama_index.core import Document, VectorStoreIndex

    prela.init(service_name="llamaindex-errors", exporter="console")

    try:
        # Create index with no documents (will fail)
        documents = []
        index = VectorStoreIndex.from_documents(documents)
        response = index.as_query_engine().query("Test query")
    except Exception as e:
        print(f"\nCaught error: {type(e).__name__}: {e}")
        print("Error was captured in the trace!")

    # Even failed operations are traced
    # The span will have status=ERROR and error details


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("LlamaIndex Instrumentation Examples")
    print("=" * 60)
    print("\nThese examples require:")
    print("  - llama-index")
    print("  - llama-index-llms-openai")
    print("  - llama-index-embeddings-openai")
    print("  - OPENAI_API_KEY environment variable")
    print("\nNote: Running these examples will make actual API calls")
    print("=" * 60)

    # Check if user wants to run examples
    if "--run" not in sys.argv:
        print("\nTo run examples: python llamaindex_instrumentation.py --run")
        print("(This will use your OpenAI API credits)")
        sys.exit(0)

    # Run examples
    try:
        example_basic_query()
        example_custom_retrieval()
        example_chat_engine()
        example_sub_question_query()
        example_with_metadata()
        example_error_handling()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("\nInstall with:")
        print(
            "  pip install llama-index llama-index-llms-openai llama-index-embeddings-openai"
        )
        sys.exit(1)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
