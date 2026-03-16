"""Feedback-driven fine-tuning pipeline for S1-M7.

Scans corrections collected by the serving layer, merges them with the
original training set, runs a fresh LoRA fine-tuning pass, and produces
a comparison report against the v2 baseline (Val WER 44.02%).
"""
