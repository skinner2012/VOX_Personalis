#!/bin/bash
# Run code quality checks and optionally fix issues
# Usage:
#   ./scripts/code_quality_check.sh                   # Check everything (no changes)
#   ./scripts/code_quality_check.sh --fix             # Check and auto-fix everything
#   ./scripts/code_quality_check.sh scripts/          # Check specific directory
#   ./scripts/code_quality_check.sh --fix scripts/    # Check and fix specific directory

FIX_MODE=false
TARGETS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --fix)
      FIX_MODE=true
      shift
      ;;
    *)
      TARGETS+=("$1")
      shift
      ;;
  esac
done

# If no targets specified, default to current directory
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(".")
fi

# Categorize targets by type
PYTHON_TARGETS=()
MD_TARGETS=()
SHELL_TARGETS=()

for target in "${TARGETS[@]}"; do
  if [[ "${target}" == *.py ]]; then
    PYTHON_TARGETS+=("${target}")
  elif [[ "${target}" == *.md ]]; then
    MD_TARGETS+=("${target}")
  elif [[ "${target}" == *.sh ]]; then
    SHELL_TARGETS+=("${target}")
  elif [[ -d "${target}" ]] || [[ "${target}" == "." ]]; then
    PYTHON_TARGETS+=("${target}")
    MD_TARGETS+=("${target}")
    SHELL_TARGETS+=("${target}")
  fi
done

echo "=== Code Quality Check for: ${TARGETS[*]} ==="
echo ""

if [[ "${FIX_MODE}" = true ]]; then
  echo "Mode: AUTO-FIX (will modify files)"
else
  echo "Mode: CHECK ONLY (no changes)"
fi
echo ""

OVERALL_EXIT=0

# Python checks
if [[ ${#PYTHON_TARGETS[@]} -gt 0 ]]; then
  echo "=== Python Files ==="

  if [[ "${FIX_MODE}" = true ]]; then
    echo "1. Auto-fixing linting issues..."
    ruff check --fix "${PYTHON_TARGETS[@]}"

    echo -e "\n2. Formatting Python code..."
    ruff format "${PYTHON_TARGETS[@]}"
  else
    echo "1. Checking linting..."
    ruff check "${PYTHON_TARGETS[@]}" || OVERALL_EXIT=1

    echo -e "\n2. Checking formatting..."
    ruff format --check "${PYTHON_TARGETS[@]}" || OVERALL_EXIT=1
  fi

  echo -e "\n3. Checking types..."
  mypy "${PYTHON_TARGETS[@]}" || OVERALL_EXIT=1
  echo ""
fi

# Markdown checks
if [[ ${#MD_TARGETS[@]} -gt 0 ]]; then
  echo "=== Markdown Files ==="

  if [[ "${FIX_MODE}" = true ]]; then
    echo "1. Formatting Markdown files..."
    mdformat "${MD_TARGETS[@]}"
  else
    echo "1. Checking Markdown formatting..."
    mdformat --check "${MD_TARGETS[@]}" || OVERALL_EXIT=1
  fi

  echo -e "\n2. Linting Markdown files..."
  pymarkdown --config .pymarkdown.json scan "${MD_TARGETS[@]}" || OVERALL_EXIT=1
  echo ""
fi

# Shell script checks
if [[ ${#SHELL_TARGETS[@]} -gt 0 ]]; then
  echo "=== Shell Scripts ==="

  # Note: shellcheck doesn't handle directories, so find .sh files if target is a directory
  SHELL_FILES=()
  for target in "${SHELL_TARGETS[@]}"; do
    if [[ -d "${target}" ]]; then
      # Find all .sh files in directory (Bash 3.2 compatible)
      while IFS= read -r file; do
        SHELL_FILES+=("${file}")
      done < <(find "${target}" -type f -name "*.sh")
    else
      SHELL_FILES+=("${target}")
    fi
  done

  if [[ ${#SHELL_FILES[@]} -gt 0 ]]; then
    echo "1. Linting shell scripts..."
    shellcheck "${SHELL_FILES[@]}" || OVERALL_EXIT=1
  fi

  if [[ "${FIX_MODE}" = true ]]; then
    echo -e "\n2. Formatting shell scripts..."
    shfmt -i 2 -bn -ci -w "${SHELL_TARGETS[@]}"
  else
    echo -e "\n2. Checking shell script formatting..."
    shfmt -i 2 -bn -ci -d "${SHELL_TARGETS[@]}" || OVERALL_EXIT=1
  fi
  echo ""
fi

# Summary
if [[ "${FIX_MODE}" = true ]]; then
  echo "=== Done! ==="
else
  echo "=== Summary ==="
  if [[ "${OVERALL_EXIT}" -eq 0 ]]; then
    echo "✓ All checks passed!"
  else
    echo "✗ Some checks failed."
    echo "Run './scripts/code_quality_check.sh --fix ${TARGETS[*]}' to auto-fix."
    exit 1
  fi
fi
