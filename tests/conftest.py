"""Suite-wide settings, applied before any test module imports beatrig."""
import os

# MCP tools tell a running preview page what they did (POST /api/notify). Point that at a closed port so a test run
# never posts fake pushes and undos into a real preview page's agent-activity feed.
os.environ["BEATRIG_PREVIEW_URL"] = "http://127.0.0.1:9"
