# ============================================================
# TASK 1: CHATBOT WITH RULE-BASED RESPONSES
# CodSoft AI Internship
# ============================================================

import re
import random
import math
from datetime import datetime

# ---------- Anti-Repeat Response Picker ----------
# Tracks last 2 responses per pattern key to avoid immediate repeats
_response_history: dict = {}

def pick_response(key: str, responses: list) -> str:
    """Pick a response that wasn't used in the last 2 turns for this key."""
    history = _response_history.get(key, [])
    available = [r for r in responses if r not in history]

    # If all responses are in history (very small list), just reset
    if not available:
        available = responses

    chosen = random.choice(available)

    # Update history, keep only last 2
    history.append(chosen)
    _response_history[key] = history[-2:]
    return chosen


# ---------- Math Engine ----------
def try_math(user_input: str):
    """
    Attempts to parse and evaluate a math expression from user input.
    Supports: + - * / ** // % sqrt log sin cos tan factorial abs
    Returns result string or None if not a math query.
    """
    text = user_input.lower().strip()

    # Replace written words with symbols/functions
    replacements = [
        (r'\bsquare root of\b',    'sqrt'),
        (r'\bsqrt of\b',           'sqrt'),
        (r'\bcube root of\b',      'cbrt'),
        (r'\bto the power of\b',   '**'),
        (r'\bpower\b',             '**'),
        (r'\bdivided by\b',        '/'),
        (r'\btimes\b',             '*'),
        (r'\bmultiplied by\b',     '*'),
        (r'\bplus\b',              '+'),
        (r'\bminus\b',             '-'),
        (r'\bmod\b',               '%'),
        (r'\bmodulo\b',            '%'),
        (r'\bfactorial of\b',      'factorial'),
        (r'\bfactorial\b',         'factorial'),
        (r'\bsin of\b',            'sin'),
        (r'\bcos of\b',            'cos'),
        (r'\btan of\b',            'tan'),
        (r'\blog of\b',            'log10'),
        (r'\bln of\b',             'log'),
        (r'\bpi\b',                str(math.pi)),
        (r'\babs\b',               'abs'),
        (r'\babsolute value of\b', 'abs'),
        (r'what is|calculate|compute|evaluate|solve', ''),
        (r'\?', ''),
    ]

    expr = text
    for pattern, replacement in replacements:
        expr = re.sub(pattern, replacement, expr)

    expr = expr.strip()

    # Must contain digits to be a math expression
    if not re.search(r'\d', expr):
        return None

    # Must contain a math operator or function
    if not re.search(r'[\+\-\*\/\%\^]|sqrt|log|sin|cos|tan|factorial|cbrt|abs', expr):
        return None

    # Safe math evaluation namespace
    safe_globals = {
        "__builtins__": {},
        "sqrt":      math.sqrt,
        "cbrt":      lambda x: x ** (1/3),
        "log":       math.log,
        "log10":     math.log10,
        "log2":      math.log2,
        "sin":       lambda x: math.sin(math.radians(x)),
        "cos":       lambda x: math.cos(math.radians(x)),
        "tan":       lambda x: math.tan(math.radians(x)),
        "factorial": math.factorial,
        "abs":       abs,
        "pi":        math.pi,
        "e":         math.e,
        "pow":       pow,
        "round":     round,
    }

    try:
        result = eval(expr, safe_globals)

        # Format result nicely
        if isinstance(result, float):
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 6)

        responses = [
            f"The answer is {result} 🧮",
            f"That comes out to {result}.",
            f"Let me crunch that... {result}!",
            f"Result: {result} ✅",
            f"Calculated: {result} 💡",
        ]
        return pick_response("math", responses)

    except ZeroDivisionError:
        return "Oops! You can't divide by zero. 😅"
    except ValueError as e:
        return f"Math error: {e}"
    except Exception:
        return None  # Not a math expression, let other rules handle it


# ---------- Response Rules ----------
RULES = {
    r'\b(hi|hello|hey|howdy)\b': [
        "Hello! How can I help you today?",
        "Hey there! What's on your mind?",
        "Hi! Great to see you!",
        "Howdy! What can I do for you?",
        "Hello hello! Ready to chat 😊",
        "Hey! Hope you're having a great day!",
        "Hi there! Ask me anything — even math! 🧮",
    ],
    r'\b(how are you|how do you do|how\'s it going)\b': [
        "I'm doing great, thanks for asking!",
        "I'm just a bot, but I'm running perfectly fine!",
        "All systems go! How about you?",
        "Feeling electric today! ⚡ You?",
    ],
    r'\b(your name|who are you|what are you)\b': [
        "I'm Sarge, your simple AI assistant!",
        "I'm a rule-based chatbot built for the CodSoft internship.",
        "Call me RuleBot — I'm here to help!"
    ],
    r'\b(bye|goodbye|see you|exit|quit)\b': [
        "Goodbye! Have a great day!",
        "See you later!",
        "Bye! Take care! 👋",
        "Until next time! 😊"
    ],
    r'\b(time|what time)\b': [
        "The current time is {time}.",
        "It's {time} right now!",
        "Clock says: {time} ⏰"
    ],
    r'\b(date|today|what day)\b': [
        "Today is {date}.",
        "The date today is {date} 📅",
        "It's {date} today!"
    ],
    r'\b(help|support|assist)\b': [
        "Sure! You can ask me about the time, date, jokes, math, or just chat!",
        "I can do math, tell jokes, share time/date, and more. Try: 'what is 25 * 4'",
        "Need help? Ask me math questions, for the time, or just say hi!"
    ],
    r'\b(joke|funny|laugh)\b': [
        "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
        "Why did the AI go to school? To improve its learning curve!",
        "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads.",
        "Why was the math book sad? It had too many problems. 😄",
        "Parallel lines have so much in common... it's a shame they'll never meet."
    ],
    r'\b(weather|temperature|forecast)\b': [
        "I don't have live weather data, but you can check weather.com!",
        "I can't check the weather, but I hope it's sunny where you are! ☀️"
    ],
    r'\b(age|how old)\b': [
        "I was just created, so I'm pretty young! Age is just a number anyway."
    ],
    r'\b(thank|thanks|thank you)\b': [
        "You're welcome!",
        "Happy to help!",
        "Anytime! 😊",
        "My pleasure!"
    ],
    r'\b(what can you do|capabilities|features)\b': [
        "I can chat, tell jokes, do math (try: 'sqrt of 144'), share time/date, and more!",
        "Ask me to calculate anything — simple sums, sqrt, sin, factorial, logs, and more!"
    ],
    r'\b(math|calculate|compute|calculator)\b': [
        "Sure! Give me a math problem. Examples:\n"
        "  → 'what is 12 * (5 + 3)'\n"
        "  → 'sqrt of 225'\n"
        "  → 'factorial of 7'\n"
        "  → 'sin of 30'\n"
        "  → '2 ** 10'"
    ],
}

DEFAULT_RESPONSES = [
    "Hmm, I'm not sure how to respond to that. Try asking something else!",
    "I didn't quite understand that. Could you rephrase?",
    "Interesting! But I don't have a response for that yet.",
    "I'm still learning! Try asking about math, the time, a joke, or just say hi.",
    "Not sure about that one — but I can calculate things! Try: '15 * 20 + 5'"
]

# ---------- Dynamic Placeholders ----------
def fill_placeholders(response: str) -> str:
    response = response.replace("{time}", datetime.now().strftime('%H:%M:%S'))
    response = response.replace("{date}", datetime.now().strftime('%A, %B %d, %Y'))
    return response

# ---------- Main Response Logic ----------
def get_response(user_input: str) -> str:
    user_input_stripped = user_input.strip()

    # 1. Try math first
    math_result = try_math(user_input_stripped)
    if math_result:
        return math_result

    user_input_lower = user_input_stripped.lower()

    # 2. Match rules with anti-repeat picker
    for pattern, responses in RULES.items():
        if re.search(pattern, user_input_lower):
            response = pick_response(pattern, responses)
            return fill_placeholders(response)

    # 3. Default fallback
    return pick_response("default", DEFAULT_RESPONSES)


# ---------- Run Chatbot ----------
def run_chatbot():
    print("=" * 52)
    print("       Welcome to RuleBot - CodSoft AI Task 1")
    print("=" * 52)
    print("💡 I can chat AND do math! Try:")
    print("   → 'what is 144 / 12'")
    print("   → 'sqrt of 256'")
    print("   → 'factorial of 6'")
    print("   → 'sin of 45'")
    print("   → '2 ** 8 + 100'")
    print("   → '(15 + 5) * 3 - 10'")
    print("\nType 'quit' or 'bye' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        response = get_response(user_input)
        print(f"Bot: {response}\n")

        if re.search(r'\b(bye|goodbye|exit|quit)\b', user_input.lower()):
            break


if __name__ == "__main__":
    run_chatbot()