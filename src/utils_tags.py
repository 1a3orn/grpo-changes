"""
Utility functions and classes for data processing.
"""

import re
from typing import List, Dict, Tuple, Optional


class TagParser:
    """
    Parser for XML-like tags in strings.
    """
    
    def __init__(self, expected_tags: List[str]):
        self.expected_tags = expected_tags
        self.tag_patterns = {tag: re.compile(f'<{tag}>(.*?)</{tag}>', re.DOTALL) for tag in expected_tags}
    
    def __call__(self, text: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        """
        Parse the text and extract content from expected tags.
        
        Args:
            text: String containing XML-like tags
            
        Returns:
            Tuple of (dict, None) on success, or (None, error_message) on failure
        """        
        # Check for unclosed tags
        if self._has_unclosed_tags(text):
            return None, "Unclosed tags detected"
        
        # Check for tags in wrong order
        if not self._tags_in_correct_order(text):
            return None, "Tags are not in the expected order"
        
        # Extract content from each tag
        result = {}
        for tag in self.expected_tags:
            match = self.tag_patterns[tag].search(text)
            if not match:
                return None, f"Missing required tag: <{tag}>"
            result[tag] = match.group(1).strip()
        
        return result, None
    
    def _has_text_outside_tags(self, text: str) -> bool:
        """Check if there's any text outside of the expected tags."""
        # Remove all expected tags and their content
        cleaned_text = text
        for tag in self.expected_tags:
            cleaned_text = re.sub(f'<{tag}>.*?</{tag}>', '', cleaned_text, flags=re.DOTALL)
        
        # Check if there's any non-whitespace text left
        return bool(re.search(r'\S', cleaned_text))
    
    def _has_unclosed_tags(self, text: str) -> bool:
        """Check for unclosed tags."""
        # Count opening and closing tags for each expected tag
        for tag in self.expected_tags:
            opening_count = len(re.findall(f'<{tag}>', text))
            closing_count = len(re.findall(f'</{tag}>', text))
            if opening_count != closing_count:
                return True
        return False
    
    def _tags_in_correct_order(self, text: str) -> bool:
        """Check if tags appear in the expected order."""
        # Find all tag positions
        tag_positions = []
        for tag in self.expected_tags:
            # Find opening tags
            for match in re.finditer(f'<{tag}>', text):
                tag_positions.append((match.start(), tag, 'open'))
            # Find closing tags
            for match in re.finditer(f'</{tag}>', text):
                tag_positions.append((match.start(), tag, 'close'))
        
        # Sort by position
        tag_positions.sort(key=lambda x: x[0])
        
        # Check if tags are properly nested and in order
        stack = []
        expected_order = []
        
        for pos, tag, tag_type in tag_positions:
            if tag_type == 'open':
                stack.append(tag)
                expected_order.append(tag)
            else:  # closing tag
                if not stack or stack.pop() != tag:
                    return False  # Improper nesting
        
        # Check if the order matches expected_tags
        return expected_order == self.expected_tags


def main():
    """Example usage of TagParser."""
    # Create parser for "think" and "answer" tags
    parser = TagParser(["think", "answer"])
    
    # Test cases
    test_cases = [
        # Valid case
        """<think>This is my reasoning</think><answer>The final answer</answer>""",
        
        # Missing tag
        """<think>This is my reasoning</think>""",
        
        # Wrong order
        """<answer>The final answer</answer><think>This is my reasoning</think>""",
        
        # Text outside tags
        """Some text<think>This is my reasoning</think><answer>The final answer</answer>""",
        
        # Unclosed tag
        """<think>This is my reasoning<answer>The final answer</answer>""",
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Input: {test_text}")
        result, error = parser(test_text)
        if result:
            print(f"Success: {result}")
        else:
            print(f"Error: {error}")


if __name__ == "__main__":
    main() 