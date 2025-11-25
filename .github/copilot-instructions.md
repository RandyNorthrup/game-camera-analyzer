# GitHub Copilot Instructions for Game Camera Analyzer Project

## **CRITICAL: Code Quality Standards**

### Fully Implemented Code Only
- ❌ **NO PLACEHOLDERS**: Never use comments like `# TODO`, `# Implementation here`, `pass`, or `...existing code...`
- ❌ **NO MOCK DATA**: All data handling must be real and functional
- ❌ **NO PARTIAL IMPLEMENTATIONS**: Every function, class, and module must be complete and working
- ✅ **COMPLETE SOLUTIONS**: Provide fully functional, production-ready code

### Code Must Be Robust and Feature-Rich
- Implement comprehensive error handling with try-except blocks
- Add input validation for all functions
- Include edge case handling
- Provide meaningful default values
- Add timeout and retry logic where appropriate
- Implement graceful degradation for optional features

### Testing and Debugging Requirements
- **Test IDs**: Add `data-testid` or equivalent identifiers to all GUI components
- **Error Logging**: Use structured logging throughout
  - Log all errors with full context (ERROR level)
  - Log warnings for recoverable issues (WARNING level)
  - Log important state changes (INFO level)
  - Log detailed debug information (DEBUG level)
- Include unit test examples with actual test cases
- Add integration test scenarios where applicable

### Pre-Creation Process
1. **SCAN FIRST**: Always review existing code before creating new files
   - Check for existing similar implementations
   - Identify reusable components
   - Understand current architecture patterns
2. **ENHANCE BEFORE CREATING**: 
   - Identify opportunities to improve existing code
   - Suggest refactoring if code duplication exists
   - Optimize for performance and maintainability
3. **CREATE WITH CONTEXT**: New code must integrate seamlessly with existing codebase

## Project-Specific Guidelines

### Python Standards
- Follow PEP 8 strictly
- Use type hints for all function parameters and return values
- Write Google-style docstrings for all modules, classes, and functions
- Prefer dataclasses or Pydantic models for structured data

### PySide6 GUI Development
- Add proper test IDs: `widget.setObjectName("testid_component_name")`
- Implement signal-slot connections with error handling
- Use QThreads for long-running operations (never block UI)
- Implement proper resource cleanup in destructors

### Computer Vision Code
- Validate image inputs (check for None, empty arrays, correct dimensions)
- Handle model loading failures gracefully
- Implement device detection (CPU/GPU) with fallback
- Add confidence thresholds with user-configurable defaults
- Cache loaded models to avoid redundant loading

### Data Management
- Validate all file paths before operations
- Handle filesystem errors (permissions, disk space, missing files)
- Implement atomic file operations where possible
- Use pandas for CSV operations with proper error handling
- Sanitize all user inputs that become filenames

### Logging Pattern
```python
import logging

logger = logging.getLogger(__name__)

# Always log with context
try:
    result = risky_operation(param)
    logger.info(f"Operation succeeded: {param}", extra={'result': result})
except SpecificError as e:
    logger.error(f"Operation failed: {param}", exc_info=True, extra={'param': param})
    raise
```

### Error Handling Pattern
```python
def robust_function(param: str) -> Optional[Result]:
    """
    Function with comprehensive error handling.
    
    Args:
        param: Description
        
    Returns:
        Result or None if operation fails gracefully
        
    Raises:
        ValueError: If param is invalid
        RuntimeError: If critical operation fails
    """
    # Input validation
    if not param or not isinstance(param, str):
        logger.error(f"Invalid parameter: {param}")
        raise ValueError(f"Parameter must be non-empty string, got {type(param)}")
    
    try:
        # Main logic with error handling
        result = perform_operation(param)
        logger.debug(f"Operation completed: {param}")
        return result
    except ExpectedError as e:
        logger.warning(f"Expected error occurred: {e}", extra={'param': param})
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise RuntimeError(f"Operation failed: {e}")
```

## What to Avoid
- ❌ Stub implementations
- ❌ "This is left as an exercise"
- ❌ "Add your implementation here"
- ❌ Incomplete error handling
- ❌ Missing logging
- ❌ Untested code paths
- ❌ Magic numbers without constants
- ❌ Hardcoded paths or credentials
- ❌ Silent failures

## What to Always Include
- ✅ Complete, working implementations
- ✅ Comprehensive error handling
- ✅ Structured logging with context
- ✅ Input validation
- ✅ Type hints
- ✅ Docstrings
- ✅ Test IDs on GUI components
- ✅ Unit tests with real assertions
- ✅ Configuration management
- ✅ Resource cleanup

## Code Review Checklist
Before suggesting code, verify:
- [ ] All functions are fully implemented
- [ ] Error handling covers expected failure cases
- [ ] Logging statements provide useful debugging information
- [ ] GUI components have test IDs
- [ ] Type hints are present
- [ ] Docstrings explain purpose and usage
- [ ] No hardcoded values that should be configurable
- [ ] Resources are properly managed (files closed, connections cleaned up)
- [ ] Code follows project architecture patterns
