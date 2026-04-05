---
description: 'Python coding conventions and guidelines'
applyTo: '**/*.py'
---

# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following PEP 257 conventions.
- Use the `typing` module for type annotations (e.g., `List[str]`, `Dict[str, int]`).
- Break down complex functions into smaller, more manageable functions.

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.

## Import Transfer Safety (Rclone)

- Follow official rclone behavior for `copy` and `move` commands: https://rclone.org/docs/
- Treat `copy` as non-destructive: source files remain after successful transfer.
- Treat `move` as destructive: source files are removed only after successful transfer.
- Use file modification time and size as the default criteria to decide whether files are the same.
- Do not use MD5/checksum matching for routine import conflict detection unless explicitly requested.
- Keep `--update` enabled for import transfers so destination-newer files are not overwritten.
- Only use `--check-first` when explicitly requested, since it can increase memory usage.
- Do not change existing overwrite and transfer-safety semantics unless explicitly requested.
- Re-imports of the same card should be idempotent and quiet when content is unchanged.
- Surface conflicts only for true overwrite risk: incoming content differs from an existing completed destination file.
- Allow overwriting partial/incomplete destination files left by failed transfers so retries can complete cleanly.

## Code Style and Formatting

- Follow the **PEP 8** style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 79 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword.
- Use blank lines to separate functions, classes, and code blocks where appropriate.

## Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases.
- Write unit tests for functions and document them with docstrings explaining the test cases.

## Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle given the radius.
    
    Parameters:
    radius (float): The radius of the circle.
    
    Returns:
    float: The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
```
