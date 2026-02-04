#!/usr/bin/env python3
"""
Example: Simple n8n webhook receiver using prela.init()

This example shows the easiest way to start an n8n webhook receiver
using the built-in n8n_webhook_port parameter in prela.init().

The webhook handler runs in a background thread and doesn't block
your main application.
"""

import prela
import time

# One-line initialization with n8n webhook receiver
tracer = prela.init(
    service_name="n8n-workflows",
    exporter="console",  # Use "http" for production
    n8n_webhook_port=8787,  # Start webhook receiver on port 8787
    verbosity="verbose",  # Show detailed trace output
)

print("✓ n8n webhook receiver started on http://0.0.0.0:8787")
print("\nConfigure your n8n workflow to POST to:")
print("  http://your-server:8787/webhook")
print("\nWebhook payload format:")
print("  {")
print('    "workflow": {{ $workflow }},')
print('    "execution": {{ $execution }},')
print('    "node": {{ $node }},')
print('    "data": {{ $json }}')
print("  }")
print("\nPress Ctrl+C to stop...")

try:
    # Keep the script running to receive webhooks
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\nShutting down webhook receiver...")
