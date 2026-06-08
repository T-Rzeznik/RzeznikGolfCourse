"""The stats chatbot: a thin LLM "mouth" over the deterministic golf "brain".

- golf.stats          -> the brain (every number lives here)
- chatbot.tools       -> Gemini function declarations + dispatch to the brain
- chatbot.prompts     -> the system prompt that forbids inventing stats
- chatbot.gemini      -> the Gemini client + tool-calling loop (answer())
- chatbot.server      -> a local FastAPI server (/api/chat) that also serves the webapp

Run it with `python run_server.py` from the repo root.
"""
