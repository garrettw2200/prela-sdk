"""
Example: Evaluating n8n Workflows with Prela

This example demonstrates how to use Prela's n8n evaluation framework to test
n8n workflows by triggering them, waiting for completion, and asserting on results.

Usage:
    python examples/n8n_eval_example.py
"""

import asyncio
import os

import prela
from prela.evals.assertions import ContainsAssertion, RegexAssertion
from prela.evals.n8n import (
    N8nEvalCase,
    N8nWorkflowEvalConfig,
    N8nWorkflowEvalRunner,
    eval_n8n_workflow,
)
from prela.evals.reporters import ConsoleReporter


# Example 1: Basic workflow evaluation
async def example_basic_workflow_eval():
    """Test a simple n8n workflow with basic assertions."""
    print("=" * 60)
    print("Example 1: Basic Workflow Evaluation")
    print("=" * 60)

    # Define test case
    case = N8nEvalCase(
        id="test_basic",
        name="Basic greeting workflow",
        trigger_data={"name": "Alice", "language": "English"},
        expected_output={"greeting": "Hello, Alice!"},
    )

    # Run evaluation
    results = await eval_n8n_workflow(
        workflow_id="greeting-workflow-123",
        test_cases=[case],
        n8n_url="http://localhost:5678",
    )

    print(f"Suite: {results['suite_name']}")
    print(f"Passed: {results['passed']}/{results['total']}")
    print()


# Example 2: Lead scoring workflow with node-level assertions
async def example_lead_scoring_eval():
    """Test a lead scoring workflow with assertions on specific nodes."""
    print("=" * 60)
    print("Example 2: Lead Scoring Workflow with Node Assertions")
    print("=" * 60)

    # Initialize Prela for trace capture
    tracer = prela.init(service_name="n8n-eval-demo", exporter="console")

    test_cases = [
        # Test case 1: High-intent lead
        N8nEvalCase(
            id="test_high_intent",
            name="High-intent lead classification",
            trigger_data={
                "email_body": "I want to buy your premium plan immediately. What are the pricing options?",
                "sender_email": "cto@acmecorp.com",
                "company_size": 500,
            },
            node_assertions={
                "AI Intent Classifier": [
                    ContainsAssertion(text="high_intent"),
                    ContainsAssertion(text="purchase"),
                ],
                "Lead Scorer": [
                    RegexAssertion(pattern=r'"score":\s*[89]\d'),  # Score 80-99
                ],
            },
            expected_output={
                "intent": "high_intent",
                "score": 95,
                "priority": "urgent",
            },
        ),
        # Test case 2: Low-intent lead
        N8nEvalCase(
            id="test_low_intent",
            name="Low-intent inquiry classification",
            trigger_data={
                "email_body": "Can you tell me more about your company?",
                "sender_email": "user@example.com",
                "company_size": 5,
            },
            node_assertions={
                "AI Intent Classifier": [
                    ContainsAssertion(text="low_intent"),
                ],
                "Lead Scorer": [
                    RegexAssertion(pattern=r'"score":\s*[0-4]\d'),  # Score 0-49
                ],
            },
            expected_output={
                "intent": "low_intent",
                "score": 25,
                "priority": "low",
            },
        ),
        # Test case 3: Technical question
        N8nEvalCase(
            id="test_technical",
            name="Technical support question",
            trigger_data={
                "email_body": "How does your API authentication work? I'm getting a 401 error.",
                "sender_email": "developer@customer.com",
                "company_size": 100,
            },
            node_assertions={
                "AI Intent Classifier": [
                    ContainsAssertion(text="technical_support"),
                ],
                "Route to Support": [
                    ContainsAssertion(text="engineering_team"),
                ],
            },
        ),
    ]

    # Configure runner
    config = N8nWorkflowEvalConfig(
        workflow_id="lead-scoring-abc123",
        n8n_base_url="http://localhost:5678",
        n8n_api_key=os.getenv("N8N_API_KEY"),
        timeout_seconds=60,
        capture_traces=True,
    )

    runner = N8nWorkflowEvalRunner(config, tracer=tracer)

    try:
        # Run all test cases
        from prela.evals.suite import EvalSuite

        suite = EvalSuite(name="Lead Scoring Tests", cases=test_cases)
        results = await runner.run_suite(suite)

        # Print detailed results
        print(f"\nSuite: {results['suite_name']}")
        print(f"Total: {results['total']}")
        print(f"Passed: {results['passed']}")
        print(f"Failed: {results['failed']}")
        print()

        for case_result in results["cases"]:
            status = "✓" if case_result["passed"] else "✗"
            print(f"{status} {case_result.get('execution_id', 'N/A')}")
            print(f"  Duration: {case_result['duration_ms']:.1f}ms")
            print(f"  Status: {case_result['status']}")

            # Show node assertion results
            if case_result["node_results"]:
                print("  Node Results:")
                for node_name, assertions in case_result["node_results"].items():
                    print(f"    {node_name}:")
                    for assertion in assertions:
                        result_status = "✓" if assertion.passed else "✗"
                        print(f"      {result_status} {assertion.message}")

            print()

    finally:
        await runner.close()


# Example 3: Customer support routing workflow
async def example_customer_support_eval():
    """Test a customer support routing workflow."""
    print("=" * 60)
    print("Example 3: Customer Support Routing Workflow")
    print("=" * 60)

    test_cases = [
        N8nEvalCase(
            id="test_billing",
            name="Billing inquiry routing",
            trigger_data={
                "subject": "Invoice question",
                "message": "Why was I charged twice this month?",
                "customer_tier": "premium",
            },
            node_assertions={
                "Classify Support Type": [
                    ContainsAssertion(text="billing"),
                ],
                "Route to Team": [
                    ContainsAssertion(text="finance_team"),
                ],
            },
        ),
        N8nEvalCase(
            id="test_bug_report",
            name="Bug report routing",
            trigger_data={
                "subject": "App crashes when uploading files",
                "message": "The app consistently crashes when I try to upload PDFs larger than 10MB.",
                "customer_tier": "enterprise",
            },
            node_assertions={
                "Classify Support Type": [
                    ContainsAssertion(text="bug"),
                ],
                "Route to Team": [
                    ContainsAssertion(text="engineering_team"),
                ],
                "Priority Escalation": [
                    ContainsAssertion(text="high_priority"),
                ],
            },
        ),
    ]

    results = await eval_n8n_workflow(
        workflow_id="support-routing-xyz789",
        test_cases=test_cases,
        n8n_url="http://localhost:5678",
        n8n_api_key=os.getenv("N8N_API_KEY"),
        timeout_seconds=45,
    )

    # Use ConsoleReporter to display results (if compatible)
    print(f"Pass rate: {results['passed']}/{results['total']}")
    for case in results["cases"]:
        print(f"  - {case.get('execution_id', 'N/A')}: {'PASSED' if case['passed'] else 'FAILED'}")
    print()


# Example 4: Content moderation workflow
async def example_content_moderation_eval():
    """Test a content moderation workflow with AI classification."""
    print("=" * 60)
    print("Example 4: Content Moderation Workflow")
    print("=" * 60)

    test_cases = [
        N8nEvalCase(
            id="test_safe_content",
            name="Safe content passes moderation",
            trigger_data={
                "content": "This is a helpful tutorial on how to bake cookies.",
                "author": "user123",
            },
            node_assertions={
                "AI Content Classifier": [
                    ContainsAssertion(text="safe"),
                    ContainsAssertion(text="approved"),
                ],
            },
            expected_output={"moderation_result": "approved", "confidence": 0.95},
        ),
        N8nEvalCase(
            id="test_spam_content",
            name="Spam content gets flagged",
            trigger_data={
                "content": "🎉 CLICK HERE TO WIN $1000000 🎉 LIMITED TIME OFFER!!!",
                "author": "spammer456",
            },
            node_assertions={
                "AI Content Classifier": [
                    ContainsAssertion(text="spam"),
                    ContainsAssertion(text="rejected"),
                ],
            },
            expected_output={"moderation_result": "rejected", "reason": "spam"},
        ),
        N8nEvalCase(
            id="test_flagged_review",
            name="Potentially harmful content flagged for review",
            trigger_data={
                "content": "Content that might need human review to determine appropriateness.",
                "author": "user789",
            },
            node_assertions={
                "AI Content Classifier": [
                    ContainsAssertion(text="review"),
                ],
                "Human Review Queue": [
                    ContainsAssertion(text="pending_review"),
                ],
            },
        ),
    ]

    results = await eval_n8n_workflow(
        workflow_id="content-moderation-def456",
        test_cases=test_cases,
        n8n_url="http://localhost:5678",
    )

    print(f"Moderation Tests: {results['passed']}/{results['total']} passed")
    print()


# Example 5: Using configuration object for advanced settings
async def example_advanced_config():
    """Demonstrate advanced configuration options."""
    print("=" * 60)
    print("Example 5: Advanced Configuration")
    print("=" * 60)

    # Create custom configuration
    config = N8nWorkflowEvalConfig(
        workflow_id="complex-workflow-789",
        n8n_base_url="https://n8n.mycompany.com",
        n8n_api_key=os.getenv("N8N_API_KEY"),
        timeout_seconds=180,  # 3 minutes for long-running workflows
        capture_traces=True,
    )

    # Initialize tracer for observability
    tracer = prela.init(service_name="n8n-eval-advanced", exporter="console")

    # Create runner
    runner = N8nWorkflowEvalRunner(config, tracer=tracer)

    # Define complex test case
    case = N8nEvalCase(
        id="test_complex",
        name="Multi-step data processing workflow",
        trigger_data={
            "dataset_url": "https://example.com/data.csv",
            "processing_steps": ["clean", "transform", "aggregate"],
            "output_format": "json",
        },
        node_assertions={
            "Data Fetcher": [ContainsAssertion(text="success")],
            "Data Cleaner": [ContainsAssertion(text="rows_cleaned")],
            "Data Transformer": [ContainsAssertion(text="transformed")],
            "Data Aggregator": [ContainsAssertion(text="aggregated")],
        },
    )

    try:
        result = await runner.run_case(case)
        print(f"Execution: {'PASSED' if result['passed'] else 'FAILED'}")
        print(f"Duration: {result['duration_ms']:.1f}ms")
        print(f"Trace ID: {result.get('trace_id')}")
        print()
    finally:
        await runner.close()


# Main entry point
async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Prela n8n Workflow Evaluation Examples")
    print("=" * 60 + "\n")

    # Note: These examples assume n8n is running locally or accessible at the URLs
    # You'll need to replace workflow_id values with actual workflow IDs from your n8n instance

    try:
        # Uncomment the examples you want to run:
        # await example_basic_workflow_eval()
        # await example_lead_scoring_eval()
        # await example_customer_support_eval()
        # await example_content_moderation_eval()
        # await example_advanced_config()

        print("\nNote: To run these examples, you need:")
        print("1. n8n running locally or accessible via URL")
        print("2. Valid workflow IDs for your n8n instance")
        print("3. n8n API key (if authentication enabled)")
        print("4. Set N8N_API_KEY environment variable if needed")
        print()
        print("Uncomment the example calls in main() to run them.")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
