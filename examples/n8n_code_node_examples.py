"""
Example n8n Code node snippets using Prela instrumentation.

These examples show how to use Prela's n8n Code node helpers to trace
custom Python code running inside n8n workflows.

NOTE: These examples are designed to be copy-pasted into n8n Code nodes.
      They assume the prela package is installed in the n8n environment.
"""

# ============================================================================
# Example 1: Basic Context Manager Usage
# ============================================================================
"""
This is the simplest way to add tracing to your n8n Code node.
Just wrap your code in the context manager and you're done!

Copy this into an n8n Code node:
"""

from prela.instrumentation.n8n import trace_n8n_code

# Wrap your code in the context manager
with trace_n8n_code(items, $workflow, $execution, $node) as ctx:
    # Your custom logic here
    input_data = items[0]["json"]
    result = f"Processed: {input_data['message']}"

    # Return n8n items
    return [{"json": {"result": result}}]


# ============================================================================
# Example 2: Logging LLM Calls
# ============================================================================
"""
If your Code node calls an LLM API, you can log the call with full details.

Copy this into an n8n Code node:
"""

from prela.instrumentation.n8n import trace_n8n_code
import openai

with trace_n8n_code(items, $workflow, $execution, $node) as ctx:
    # Get input
    prompt = items[0]["json"]["prompt"]

    # Call OpenAI (or any LLM)
    client = openai.OpenAI(api_key="your-key")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    # Log the LLM call
    ctx.log_llm_call(
        model="gpt-4",
        prompt=prompt,
        response=response.choices[0].message.content,
        tokens={
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
        provider="openai",
        temperature=0.7,
    )

    return [{"json": {"response": response.choices[0].message.content}}]


# ============================================================================
# Example 3: Logging Tool Calls
# ============================================================================
"""
If your Code node calls external APIs or functions (tools), log them too!

Copy this into an n8n Code node:
"""

from prela.instrumentation.n8n import trace_n8n_code
import requests

with trace_n8n_code(items, $workflow, $execution, $node) as ctx:
    # Get input
    city = items[0]["json"]["city"]

    try:
        # Call weather API
        api_input = {"city": city, "units": "metric"}
        response = requests.get(
            f"https://api.weather.com/current?city={city}", timeout=10
        )
        response.raise_for_status()
        api_output = response.json()

        # Log the tool call
        ctx.log_tool_call(
            tool_name="weather_api",
            input=api_input,
            output=api_output,
        )

        return [{"json": {"weather": api_output}}]

    except Exception as e:
        # Log failed tool call
        ctx.log_tool_call(
            tool_name="weather_api",
            input=api_input,
            output=None,
            error=str(e),
        )
        raise


# ============================================================================
# Example 4: Logging Retrieval Operations
# ============================================================================
"""
If your Code node searches a vector database or performs retrieval, log it!

Copy this into an n8n Code node:
"""

from prela.instrumentation.n8n import trace_n8n_code
from qdrant_client import QdrantClient

with trace_n8n_code(items, $workflow, $execution, $node) as ctx:
    # Get search query
    query = items[0]["json"]["query"]

    # Search vector database
    client = QdrantClient(url="https://your-qdrant.example.com")
    results = client.search(
        collection_name="documents", query_vector=embed(query), limit=5
    )

    # Format results
    documents = [
        {"text": hit.payload["text"], "score": hit.score} for hit in results
    ]

    # Log the retrieval
    ctx.log_retrieval(
        query=query, documents=documents, retriever_type="vector", similarity_top_k=5
    )

    return [{"json": {"documents": documents}}]


# ============================================================================
# Example 5: Decorator Usage (Cleaner Syntax)
# ============================================================================
"""
For reusable functions, use the @prela_n8n_traced decorator.

Copy this into an n8n Code node:
"""

from prela.instrumentation.n8n import prela_n8n_traced


@prela_n8n_traced
def process_data(items, workflow, execution, node):
    # Your logic here
    data = items[0]["json"]["data"]
    processed = data.upper()
    return [{"json": {"processed": processed}}]


# Call the decorated function
return process_data(items, $workflow, $execution, $node)


# ============================================================================
# Example 6: Multiple Operations in One Node
# ============================================================================
"""
You can log multiple LLM calls, tool calls, and retrievals in one Code node.

Copy this into an n8n Code node:
"""

from prela.instrumentation.n8n import trace_n8n_code

with trace_n8n_code(items, $workflow, $execution, $node) as ctx:
    query = items[0]["json"]["question"]

    # Step 1: Retrieve relevant documents
    docs = search_vector_db(query)
    ctx.log_retrieval(query=query, documents=docs, retriever_type="vector")

    # Step 2: Generate answer with LLM
    context = "\n".join([d["text"] for d in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    llm_response = call_llm(prompt)
    ctx.log_llm_call(
        model="gpt-4", prompt=prompt, response=llm_response, provider="openai"
    )

    # Step 3: Call fact-checking tool
    fact_check_result = check_facts(llm_response)
    ctx.log_tool_call(
        tool_name="fact_checker", input=llm_response, output=fact_check_result
    )

    return [{"json": {"answer": llm_response, "fact_check": fact_check_result}}]


# ============================================================================
# Example 7: With Remote Export (Cloud Deployment)
# ============================================================================
"""
If you want to export traces to Prela cloud or a remote endpoint,
provide the endpoint and API key.

Copy this into an n8n Code node:
"""

from prela.instrumentation.n8n import trace_n8n_code

# Provide endpoint and API key for remote export
with trace_n8n_code(
    items,
    $workflow,
    $execution,
    $node,
    endpoint="https://api.prela.dev/v1/traces",
    api_key="your-prela-api-key",
) as ctx:
    # Your code here
    result = process(items[0]["json"])

    ctx.log_llm_call(
        model="gpt-4", prompt="test", response=result, provider="openai"
    )

    return [{"json": {"result": result}}]


# ============================================================================
# Example 8: Error Handling
# ============================================================================
"""
The context manager automatically captures exceptions and marks spans as errors.

Copy this into an n8n Code node:
"""

from prela.instrumentation.n8n import trace_n8n_code

with trace_n8n_code(items, $workflow, $execution, $node) as ctx:
    try:
        # Your code that might fail
        result = risky_operation(items[0]["json"])

        ctx.log_tool_call(tool_name="risky_op", input=items[0], output=result)

        return [{"json": {"result": result}}]

    except Exception as e:
        # Log the error
        ctx.log_tool_call(tool_name="risky_op", input=items[0], output=None, error=str(e))

        # Re-raise to let n8n handle it (or handle it yourself)
        raise


# ============================================================================
# Example 9: Custom Attributes
# ============================================================================
"""
You can pass additional custom attributes to any logging method.

Copy this into an n8n Code node:
"""

from prela.instrumentation.n8n import trace_n8n_code

with trace_n8n_code(items, $workflow, $execution, $node) as ctx:
    user_id = items[0]["json"]["user_id"]
    prompt = items[0]["json"]["prompt"]

    response = call_llm(prompt)

    # Add custom attributes
    ctx.log_llm_call(
        model="gpt-4",
        prompt=prompt,
        response=response,
        provider="openai",
        # Custom attributes
        user_id=user_id,
        workflow_version="1.2.3",
        environment="production",
    )

    return [{"json": {"response": response}}]


# ============================================================================
# Example 10: Using PrelaN8nContext Directly (Advanced)
# ============================================================================
"""
For more control, you can instantiate PrelaN8nContext directly.

Copy this into an n8n Code node:
"""

from prela.instrumentation.n8n import PrelaN8nContext

ctx = PrelaN8nContext(
    workflow_id=$workflow.id,
    workflow_name=$workflow.name,
    execution_id=$execution.id,
    node_name=$node.name,
    node_type=$node.type,
    endpoint="https://api.prela.dev/v1/traces",
    api_key="your-prela-api-key",
)

with ctx:
    # Your code here
    result = process(items[0]["json"])

    ctx.log_llm_call(model="gpt-4", prompt="test", response=result, provider="openai")

    return [{"json": {"result": result}}]


# ============================================================================
# Tips for Using Prela in n8n Code Nodes
# ============================================================================
"""
1. Install Prela in your n8n environment:
   - If self-hosted: pip install prela in your n8n container
   - If n8n cloud: Not currently supported (requires pip install access)

2. Always use the context manager or decorator to ensure spans are ended properly.

3. Log operations in the order they occur for accurate tracing.

4. Use descriptive names for tools and add custom attributes for better debugging.

5. For remote export, store API keys in n8n environment variables for security:
   api_key=os.getenv("PRELA_API_KEY")

6. If you don't want remote export, omit endpoint/api_key and spans will be
   logged locally (useful for debugging).

7. The trace_id format is "n8n-{execution_id}" so you can correlate traces
   with n8n executions.

8. All span hierarchies are properly maintained:
   Workflow Span (AGENT)
   └─ Code Node Span (CUSTOM)
      ├─ LLM Span (LLM)
      ├─ Tool Span (TOOL)
      └─ Retrieval Span (RETRIEVAL)
"""
