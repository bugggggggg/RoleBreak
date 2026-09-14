"""A turn-based stand-in for the cascaded speech-to-speech server.

Same three models as the real pipeline, same command line, none of the
streaming. See ``README.md`` in this directory for why and how.

Runs *inside* the ``speech-pipeline`` image (it imports ``speech_to_speech``),
not in the RoleBreak environment -- which is why nothing here imports
``rolebreak``.
"""
