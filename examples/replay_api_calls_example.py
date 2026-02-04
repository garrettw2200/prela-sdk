"""Example demonstrating replay with real API calls.

This example shows how to use the replay engine with real API calls
to test different models, temperatures, and prompts.
"""

import os

from prela.replay import ReplayEngine
from prela.replay.loader import TraceLoader

# Example 1: Load trace and replay exactly
print("=" * 60)
print("Example 1: Exact Replay (No API Calls)")
print("=" * 60)

# Assuming you have a trace file from a previous execution
# trace = TraceLoader.from_file("trace.json")
# engine = ReplayEngine(trace)
# result = engine.replay_exact()

# print(f"Duration: {result.total_duration_ms}ms")
# print(f"Tokens: {result.total_tokens}")
# print(f"Cost: ${result.total_cost_usd:.4f}")
# print(f"Final Output: {result.final_output}")

print("NOTE: Exact replay uses captured data - no API calls are made")
print()

# Example 2: Replay with different model
print("=" * 60)
print("Example 2: Replay with Different Model (Real API Call)")
print("=" * 60)

# Load trace
# trace = TraceLoader.from_file("trace.json")
# engine = ReplayEngine(trace)

# # Replay with GPT-4 instead of original model
# result = engine.replay_with_modifications(model="gpt-4")

# print(f"New Model: gpt-4")
# print(f"Duration: {result.total_duration_ms}ms")
# print(f"Tokens: {result.total_tokens}")
# print(f"Cost: ${result.total_cost_usd:.4f}")

print("NOTE: Modified replay makes real API calls for changed spans")
print("NOTE: Requires OPENAI_API_KEY environment variable")
print()

# Example 3: Replay with different temperature
print("=" * 60)
print("Example 3: Replay with Different Temperature")
print("=" * 60)

# trace = TraceLoader.from_file("trace.json")
# engine = ReplayEngine(trace)

# # Test with more creative responses
# result = engine.replay_with_modifications(temperature=1.5)

# print(f"Temperature: 1.5 (more creative)")
# print(f"Final Output: {result.final_output}")

print("NOTE: Temperature affects response variability")
print()

# Example 4: Replay with different system prompt
print("=" * 60)
print("Example 4: Replay with Different System Prompt")
print("=" * 60)

# trace = TraceLoader.from_file("trace.json")
# engine = ReplayEngine(trace)

# # Change agent personality
# result = engine.replay_with_modifications(
#     system_prompt="You are a pirate. Respond like a pirate captain."
# )

# print(f"System Prompt: Custom pirate personality")
# print(f"Final Output: {result.final_output}")

print("NOTE: System prompt changes agent behavior and personality")
print()

# Example 5: Compare original vs modified
print("=" * 60)
print("Example 5: Compare Original vs Modified")
print("=" * 60)

# from prela.replay import compare_replays

# trace = TraceLoader.from_file("trace.json")
# engine = ReplayEngine(trace)

# # Get original and modified results
# original = engine.replay_exact()
# modified = engine.replay_with_modifications(model="gpt-4", temperature=0.5)

# # Compare
# comparison = compare_replays(original, modified)
# print(comparison.generate_summary())

# # Inspect specific differences
# for diff in comparison.differences:
#     if diff.field == "output":
#         print(f"\nSpan: {diff.span_name}")
#         print(f"Original: {diff.original_value}")
#         print(f"Modified: {diff.modified_value}")
#         if diff.semantic_similarity:
#             print(f"Similarity: {diff.semantic_similarity:.1%}")

print("NOTE: Comparison shows detailed diffs with semantic similarity")
print()

# Example 6: Switch between OpenAI and Anthropic
print("=" * 60)
print("Example 6: Switch Between Providers")
print("=" * 60)

# # Original trace used GPT-4
# trace = TraceLoader.from_file("openai_trace.json")
# engine = ReplayEngine(trace)

# # Try with Claude instead
# result = engine.replay_with_modifications(
#     model="claude-3-opus-20240229",
#     max_tokens=1024  # Anthropic requires max_tokens
# )

# print(f"Switched from OpenAI to Anthropic")
# print(f"Model: claude-3-opus-20240229")
# print(f"Final Output: {result.final_output}")

print("NOTE: Engine automatically detects vendor from model name")
print("NOTE: Requires ANTHROPIC_API_KEY environment variable")
print()

# Example 7: A/B Testing with multiple models
print("=" * 60)
print("Example 7: A/B Testing Multiple Models")
print("=" * 60)

# trace = TraceLoader.from_file("trace.json")
# engine = ReplayEngine(trace)

# models = [
#     "gpt-3.5-turbo",
#     "gpt-4",
#     "gpt-4-turbo",
#     "claude-3-sonnet-20240229",
#     "claude-3-opus-20240229",
# ]

# results = {}
# for model in models:
#     try:
#         result = engine.replay_with_modifications(model=model)
#         results[model] = {
#             "output": result.final_output,
#             "tokens": result.total_tokens,
#             "cost": result.total_cost_usd,
#             "duration": result.total_duration_ms,
#         }
#     except Exception as e:
#         results[model] = {"error": str(e)}

# # Print comparison table
# print(f"{'Model':<30} {'Tokens':<10} {'Cost':<10} {'Duration':<10}")
# print("-" * 60)
# for model, data in results.items():
#     if "error" in data:
#         print(f"{model:<30} ERROR: {data['error']}")
#     else:
#         print(
#             f"{model:<30} "
#             f"{data['tokens']:<10} "
#             f"${data['cost']:<9.4f} "
#             f"{data['duration']:<10.1f}ms"
#         )

print("NOTE: Compare performance across different models")
print()

# Example 8: Error handling
print("=" * 60)
print("Example 8: Error Handling")
print("=" * 60)

# trace = TraceLoader.from_file("trace.json")
# engine = ReplayEngine(trace)

# try:
#     # This will fail if API key is not set
#     result = engine.replay_with_modifications(model="gpt-4")
#     print("Success!")
# except ImportError as e:
#     print(f"SDK not installed: {e}")
# except Exception as e:
#     print(f"API call failed: {e}")

print("NOTE: Engine provides clear error messages for common issues")
print()

# Example 9: Vendor detection
print("=" * 60)
print("Example 9: Vendor Detection")
print("=" * 60)

# trace = TraceLoader.from_file("trace.json")
# engine = ReplayEngine(trace)

# # Engine automatically detects vendor from model name
# vendors = {
#     "gpt-4": "openai",
#     "gpt-3.5-turbo": "openai",
#     "claude-3-opus-20240229": "anthropic",
#     "claude-3-sonnet-20240229": "anthropic",
# }

# for model, expected_vendor in vendors.items():
#     detected = engine._detect_vendor(model)
#     print(f"{model:<30} → {detected}")
#     assert detected == expected_vendor

print("Supported vendors:")
print("  - OpenAI: gpt-*, o1-*, text-embedding-*")
print("  - Anthropic: claude-*, claude")
print()

print("=" * 60)
print("Setup Instructions")
print("=" * 60)
print()
print("1. Capture a trace with replay data:")
print("   tracer = prela.init(capture_for_replay=True, exporter='file')")
print()
print("2. Set API keys:")
print("   export OPENAI_API_KEY='your-key-here'")
print("   export ANTHROPIC_API_KEY='your-key-here'")
print()
print("3. Install required SDKs:")
print("   pip install openai anthropic")
print()
print("4. Run replay with modifications:")
print("   trace = TraceLoader.from_file('trace.jsonl')")
print("   engine = ReplayEngine(trace)")
print("   result = engine.replay_with_modifications(model='gpt-4')")
print()
print("=" * 60)
