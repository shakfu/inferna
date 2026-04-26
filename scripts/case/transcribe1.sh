#!/usr/bin/env sh

# Run this from the root of inferna

uv run inferna transcribe \
	-f tests/samples/jfk.wav \
	-m models/ggml-base.en.bin