"""
Utility function for comparing numeric strings with possible $ signs, commas, and float/integer formats.
"""

def compare_strings(a: str, b: str) -> bool:
    """
    Compare two strings as numbers, ignoring $ signs and commas.
    Handles both integers and floats.
    
    Args:
        a: First string
        b: Second string
    
    Returns:
        True if numerically equal, False otherwise
    """
    def normalize(s: str):
        s = s.replace('$', '').replace(',', '').strip()
        try:
            if '.' in s:
                return float(s)
            else:
                return int(s)
        except ValueError:
            raise ValueError(f"Cannot convert '{s}' to a number.")
    
    # First try comparing with an "eval"
    try:
        eval_equals = eval(a) == eval(b)
        if eval_equals:
            return True
    except Exception as e:
        pass
    
    # Then try comparing with a "normalize"

    try:
        return normalize(a) == normalize(b)
    except ValueError:
        return False


def main():
    # Example usages
    test_cases = [
        ("$1,000", "1000"),
        ("2,500.00", "$2,500"),
        ("100", "100.0"),
        ("$3,000.50", "3000.5"),
        ("$1,000", "1,001"),
        ("abc", "100"),
        ("[1, 2, 3]", "[1, 2, 3]"),
        ('["1", "2", "3"]', "['1', '2', '3']"),
    ]
    for a, b in test_cases:
        print(f"compare_numeric_strings({a!r}, {b!r}) = {compare_numeric_strings(a, b)}")


if __name__ == "__main__":
    main() 