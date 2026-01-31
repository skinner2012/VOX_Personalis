#!/bin/bash
# Run code quality checks and optionally fix issues
# Usage:
#   ./scripts/check.sh                   # Check everything (no changes)
#   ./scripts/check.sh --fix             # Check and auto-fix everything
#   ./scripts/check.sh scripts/          # Check specific directory
#   ./scripts/check.sh --fix scripts/    # Check and fix specific directory

FIX_MODE=false
TARGET="."

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

echo "=== Code Quality Check for: $TARGET ==="
echo ""

if [ "$FIX_MODE" = true ]; then
  echo "Mode: AUTO-FIX (will modify files)"
  echo ""

  echo "1. Auto-fixing linting issues..."
  ruff check --fix "$TARGET"

  echo -e "\n2. Formatting code..."
  ruff format "$TARGET"

  echo -e "\n3. Checking types..."
  mypy "$TARGET"

  echo -e "\n=== Done! ==="
else
  echo "Mode: CHECK ONLY (no changes)"
  echo ""

  echo "1. Checking linting..."
  ruff check "$TARGET"
  LINT_EXIT=$?

  echo -e "\n2. Checking formatting..."
  ruff format --check "$TARGET"
  FORMAT_EXIT=$?

  echo -e "\n3. Checking types..."
  mypy "$TARGET"
  TYPE_EXIT=$?

  echo -e "\n=== Summary ==="
  if [ $LINT_EXIT -eq 0 ] && [ $FORMAT_EXIT -eq 0 ] && [ $TYPE_EXIT -eq 0 ]; then
    echo "✓ All checks passed!"
    exit 0
  else
    echo "✗ Some checks failed."
    echo "Run './scripts/check.sh --fix $TARGET' to auto-fix."
    exit 1
  fi
fi
