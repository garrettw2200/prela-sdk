"""
Demo: Auto-Instrumentation with Multi-Agent Frameworks

This example demonstrates that prela.init() now automatically detects and
instruments multi-agent frameworks (CrewAI, AutoGen, LangGraph, Swarm) along
with LLM SDKs (OpenAI, Anthropic) and agent frameworks (LangChain, LlamaIndex).
"""

import prela

# One-line initialization - auto-detects all installed frameworks
print("Initializing Prela with auto-instrumentation...")
tracer = prela.init(
    service_name="multi-agent-demo",
    exporter="console",  # Pretty-print traces to console
    auto_instrument=True,  # Default: True
)

print("\n✅ Prela initialized successfully!")
print("\nAuto-instrumentation will detect and instrument:")
print("  📦 LLM SDKs:")
print("    - OpenAI (if openai package installed)")
print("    - Anthropic (if anthropic package installed)")
print("\n  🤖 Agent Frameworks:")
print("    - LangChain (if langchain-core package installed)")
print("    - LlamaIndex (if llama-index-core package installed)")
print("\n  👥 Multi-Agent Frameworks:")
print("    - CrewAI (if crewai package installed)")
print("    - AutoGen (if autogen package installed)")
print("    - LangGraph (if langgraph package installed)")
print("    - Swarm (if swarm package installed)")

# Test registry access
from prela.instrumentation.auto import INSTRUMENTORS, PACKAGE_DETECTION

print(f"\n📋 Total instrumentors in registry: {len(INSTRUMENTORS)}")
print("\nRegistry contents:")
for name in INSTRUMENTORS.keys():
    pkg = PACKAGE_DETECTION.get(name, name)
    print(f"  - {name:12} (detects package: {pkg})")

print("\n💡 Usage Example:")
print("   All framework calls are now automatically traced!")
print("   No manual wrapping needed.")
print("\n   # OpenAI example")
print("   from openai import OpenAI")
print("   client = OpenAI()")
print("   response = client.chat.completions.create(...)  # ← Traced!")
print("\n   # CrewAI example")
print("   from crewai import Crew, Agent, Task")
print("   crew = Crew(agents=[...], tasks=[...])")
print("   result = crew.kickoff()  # ← Traced!")
print("\n   # LangGraph example")
print("   from langgraph.graph import StateGraph")
print("   graph = StateGraph(...)")
print("   compiled = graph.compile()")
print("   result = compiled.invoke(...)  # ← Traced!")

print("\n🎉 All instrumented operations will appear in the console exporter output")
