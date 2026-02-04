#!/bin/bash
# Run code quality checks and optionally fix issues
# Usage:
#   ./scripts/code_quality_check.sh                   # Check everything (no changes)
#   ./scripts/code_quality_check.sh --fix             # Check and auto-fix everything
#   ./scripts/code_quality_check.sh scripts/          # Check specific directory
#   ./scripts/code_quality_check.sh --fix scripts/    # Check and fix specific directory

FIX_MODE=false
TARGET="."
PYTHON_TARGET=""
MD_TARGET=""
SHELL_TARGET=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --fix)
      FIX_MODE=true
      shift
      ;;
    *)
      TARGET="$1"
      shift
      ;;
  esac
done

# Determine if we're checking Python, Markdown, or Shell files
if [[ "$TARGET" == *.py ]]; then
  PYTHON_TARGET="$TARGET"
elif [[ "$TARGET" == *.md ]]; then
  MD_TARGET="$TARGET"
elif [[ "$TARGET" == *.sh ]]; then
  SHELL_TARGET="$TARGET"
elif [[ -d "$TARGET" ]] || [[ "$TARGET" == "." ]]; then
  PYTHON_TARGET="$TARGET"
  MD_TARGET="$TARGET"
  SHELL_TARGET="$TARGET"
fi

echo "=== Code Quality Check for: $TARGET ==="
echo ""

if [ "$FIX_MODE" = true ]; then
  echo "Mode: AUTO-FIX (will modify files)"
else
  echo "Mode: CHECK ONLY (no changes)"
fi
echo ""

OVERALL_EXIT=0

# Python checks
if [ -n "$PYTHON_TARGET" ]; then
  echo "=== Python Files ==="

  if [ "$FIX_MODE" = true ]; then
    echo "1. Auto-fixing linting issues..."
    ruff check --fix "$PYTHON_TARGET"

    echo -e "\n2. Formatting Python code..."
    ruff format "$PYTHON_TARGET"
  else
    echo "1. Checking linting..."
    ruff check "$PYTHON_TARGET" || OVERALL_EXIT=1

    echo -e "\n2. Checking formatting..."
    ruff format --check "$PYTHON_TARGET" || OVERALL_EXIT=1
  fi

  echo -e "\n3. Checking types..."
  mypy "$PYTHON_TARGET" || OVERALL_EXIT=1
  echo ""
fi

# Markdown checks
if [ -n "$MD_TARGET" ]; then
  echo "=== Markdown Files ==="

  if [ "$FIX_MODE" = true ]; then
    echo "1. Formatting Markdown files..."
    mdformat "$MD_TARGET"
  else
    echo "1. Checking Markdown formatting..."
    mdformat --check "$MD_TARGET" || OVERALL_EXIT=1
  fi

  echo -e "\n2. Linting Markdown files..."
  pymarkdown --config .pymarkdown.json scan "$MD_TARGET" || OVERALL_EXIT=1
  echo ""
fi

# Shell script checks
if [ -n "$SHELL_TARGET" ]; then
  echo "=== Shell Scripts ==="

  echo "1. Linting shell scripts..."
  shellcheck "$SHELL_TARGET" || OVERALL_EXIT=1

  if [ "$FIX_MODE" = true ]; then
    echo -e "\n2. Formatting shell scripts..."
    shfmt -i 2 -bn -ci -w "$SHELL_TARGET"
  else
    echo -e "\n2. Checking shell script formatting..."
    shfmt -i 2 -bn -ci -d "$SHELL_TARGET" || OVERALL_EXIT=1
  fi
  echo ""
fi

# Summary
if [ "$FIX_MODE" = true ]; then
  echo "=== Done! ==="
else
  echo "=== Summary ==="
  if [ $OVERALL_EXIT -eq 0 ]; then
    echo "✓ All checks passed!"
  else
    echo "✗ Some checks failed."
    echo "Run './scripts/code_quality_check.sh --fix $TARGET' to auto-fix."
    exit 1
  fi
fi
