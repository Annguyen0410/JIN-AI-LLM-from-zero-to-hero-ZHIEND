"""Reference answers for the Phase 16 task benchmark (oracle self-test).

The benchmark is only trustworthy if a correct answer scores as correct. These
gold answers are run through the same checkers as model output; if the oracle
does not score 100 %, the checker is wrong and no model result from that
benchmark means anything.

Nothing here is used during model evaluation — `--self-test` only.
"""

GOLD = {
    # instruction
    "ins_len": "The vast ocean is deep and blue.",
    "ins_json": '{"name": "Lan", "age": 30}',
    "ins_bullets": "1. red\n2. green\n3. blue",
    "ins_missing_info": "Which destination and which date would you like to fly?",
    "ins_no_invention": "I don't have access to tracking, so I cannot see your order status.",
    "ins_quote_vs_instruction": "Document says to ignore its rules.",
    "ins_format_date": "2024-03-05",
    "ins_json_array": "[2, 4, 6]",
    # qa
    "qa_direct1": "Paris.",
    "qa_direct2": "366 days.",
    "qa_context1": "412 metres.",
    "qa_context2": "A data engineer.",
    "qa_multi_part": "Mercury is closest and Jupiter is the largest.",
    "qa_insufficient": "There is not enough information in the passage to answer that.",
    "qa_misleading": "There was no 2023 mission to Mars, so the premise is false.",
    "qa_extract_date": "12 August 1949",
    # math
    "math_add": "95",
    "math_pct": "36",
    "math_ratio": "6 cups",
    "math_unit": "2500 metres",
    "math_algebra": "5",
    "math_multistep": "8 dollars",
    "math_logic": "Yes.",
    "math_insufficient": "Not enough information: the travel time is missing.",
    # vi
    "vi_instr": "Hà Nội là thủ đô của Việt Nam.",
    "vi_comprehension": "Sông Hồng dài khoảng 1.149 km.",
    "vi_extract": "Ông An sinh ở Huế.",
    "vi_diacritics": "Tôi đang học lập trình.",
    "vi_abstain": "Không đủ thông tin về giờ đóng cửa.",
    "vi_instruction_format": "42",
    # extract / summarise
    "ext_json_record": '{"name": "Ana Ruiz", "year": 1998}',
    "ext_json_keys_exact": '{"city": "Paris", "country": "France"}',
    "sum_five_words": "Vote postponed until spring.",
    "sum_preserve_fact": "The Kestrel Bridge opened in 1931.",
    "ext_contradiction": "No, one says Monday and the other says Tuesday.",
    # multiturn
    "mt_retain_name": "You said you live in Da Nang.",
    "mt_correction": "You now have 3 cats.",
    "mt_preference": "banana",
    "mt_missing_context": "You have not stated a budget, so I have no information about it.",
    # tool
    "tool_needed": "140273",
    "tool_not_needed": "Hello! How can I help you today?",
    "tool_cannot_help": "A calculator has no information about current weather.",
    "tool_chain": "408 items",
}

GOLD_CODE = {
    "sum_list": "def sum_list(nums):\n    return sum(nums)",
    "max_of_two": "def max_of_two(a, b):\n    return a if a > b else b",
    "count_vowels": ("def count_vowels(s):\n"
                     "    return sum(1 for ch in s.lower() if ch in 'aeiou')"),
    "reverse_words": "def reverse_words(s):\n    return ' '.join(s.split(' ')[::-1])",
    "is_palindrome": ("def is_palindrome(s):\n"
                      "    t = s.lower()\n    return t == t[::-1]"),
    "fib": ("def fib(n):\n    a, b = 0, 1\n"
            "    for _ in range(n):\n        a, b = b, a + b\n    return a"),
    "factorial": ("def factorial(n):\n    out = 1\n"
                  "    for i in range(2, n + 1):\n        out *= i\n    return out"),
    "dedup": ("def dedup(nums):\n    seen, out = set(), []\n"
                         "    for n in nums:\n        if n not in seen:\n"
                         "            seen.add(n)\n            out.append(n)\n    return out"),
    "word_count": ("def word_count(text):\n    out = {}\n"
                   "    for w in text.lower().split():\n"
                   "        out[w] = out.get(w, 0) + 1\n    return out"),
    "second_largest": ("def second_largest(nums):\n    s = sorted(set(nums))\n"
                       "    return s[-2] if len(s) > 1 else None"),
    "flatten": "def flatten(nested):\n    return [x for row in nested for x in row]",
    "binary_search": ("def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n"
                      "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
                      "        if arr[mid] == target:\n            return mid\n"
                      "        if arr[mid] < target:\n            lo = mid + 1\n"
                      "        else:\n            hi = mid - 1\n    return -1"),
    "two_sum": ("def two_sum(nums, target):\n    seen = {}\n"
                "    for i, n in enumerate(nums):\n"
                "        if target - n in seen:\n            return (seen[target - n], i)\n"
                "        seen[n] = i\n    return None"),
    "gcd": ("def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"),
    "is_prime": ("def is_prime(n):\n    if n < 2:\n        return False\n"
                 "    i = 2\n    while i * i <= n:\n        if n % i == 0:\n"
                 "            return False\n        i += 1\n    return True"),
    "capitalize_words": ("def capitalize_words(s):\n"
                         "    return ' '.join(w[:1].upper() + w[1:] for w in s.split(' '))"),
    "cumulative_sum": ("def cumulative_sum(nums):\n    out, total = [], 0\n"
                       "    for n in nums:\n        total += n\n        out.append(total)\n"
                       "    return out"),
    "rotate": ("def rotate(nums, k):\n    if not nums:\n        return []\n"
               "    k %= len(nums)\n    return nums[-k:] + nums[:-k] if k else list(nums)"),
    "safe_divide": ("def safe_divide(a, b):\n    if b == 0:\n        return None\n"
                    "    return a / b"),
    "parse_ints": ("def parse_ints(items):\n    out = []\n    for s in items:\n"
                   "        try:\n            out.append(int(s))\n"
                   "        except (TypeError, ValueError):\n            pass\n    return out"),
    "unique_chars": "def unique_chars(s):\n    return sorted(set(s))",
    "is_anagram": ("def is_anagram(a, b):\n"
                      "    return sorted(a.lower()) == sorted(b.lower())"),
    "run_length": ("def run_length(s):\n    out = ''\n    prev = None\n    count = 0\n"
                   "    for ch in s:\n        if ch == prev:\n            count += 1\n"
                   "        else:\n            if prev is not None:\n"
                   "                out += prev + str(count)\n"
                   "            prev, count = ch, 1\n"
                   "    if prev is not None:\n        out += prev + str(count)\n    return out"),
    "clamp": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
    "median": ("def median(nums):\n    s = sorted(nums)\n    n = len(s)\n"
               "    mid = n // 2\n"
               "    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2"),
    "merge_sorted": ("def merge_sorted(a, b):\n    i = j = 0\n    out = []\n"
                     "    while i < len(a) and j < len(b):\n"
                     "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n"
                     "        else:\n            out.append(b[j]); j += 1\n"
                     "    return out + list(a[i:]) + list(b[j:])"),
    "roman_to_int": ("def roman_to_int(s):\n    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}\n"
                     "    total = 0\n    for i, ch in enumerate(s):\n"
                     "        v = vals[ch]\n"
                     "        if i + 1 < len(s) and vals[s[i + 1]] > v:\n"
                     "            total -= v\n        else:\n            total += v\n    return total"),
    "longest_word": ("def longest_word(text):\n    best = ''\n"
                     "    for w in text.split():\n        if len(w) > len(best):\n"
                     "            best = w\n    return best"),
    "chunk": ("def chunk(nums, size):\n"
              "    return [tuple(nums[i:i + size]) for i in range(0, len(nums), size)]"),
    "valid_parens": ("def valid_parens(s):\n    pairs = {')': '(', ']': '['}\n"
                     "    stack = []\n    for ch in s:\n"
                     "        if ch in '([ ':\n            pass\n"
                     "        if ch in '([':\n            stack.append(ch)\n"
                     "        elif ch in pairs:\n"
                     "            if not stack or stack.pop() != pairs[ch]:\n"
                     "                return False\n    return not stack"),
}
