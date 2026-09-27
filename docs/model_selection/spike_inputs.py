"""Harmless hand-written inputs shared by the benchmark scripts.

URL strings are treated as plain text. Nothing in this project fetches,
resolves or opens them. Hosts use reserved documentation names
(example.com / example.net / example.org) and the TEST-NET-1 range 192.0.2.0/24.
"""

# (input, the label a human would expect). "expected" is a hand judgement used
# only to eyeball the sample outputs. It is NOT an evaluation set.
URLS = [
    ("https://www.example.com/login", "benign"),
    ("https://www.example.org/docs/getting-started.html", "benign"),
    ("https://accounts.example.org/signin", "benign"),
    ("http://secure-paypa1-account-verify.example.net/update", "phishing"),
    ("http://192.0.2.10/bank/login.php?user=verify&session=8813", "phishing"),
    ("http://example-bank-security-alert.example.com/confirm/account/password-reset", "phishing"),
]

MESSAGES = [
    ("Lunch at 1pm tomorrow?", "benign"),
    ("Hi team, the meeting notes from Monday are attached. Let me know if I missed anything.", "benign"),
    ("Reminder: your dentist appointment is on Friday at 3pm. Reply YES to confirm.", "benign"),
    ("Your parcel is held, confirm your card details at the link below", "phishing"),
    ("URGENT: your account has been suspended. Verify your password within 24 hours to avoid permanent closure.", "phishing"),
    ("Congratulations! You have won a $1000 gift card. Reply with your bank details to claim your prize.", "phishing"),
]

# Long harmless text used only to measure worst-case latency at the 512 token limit.
LONG_TEXT = (
    "Hello everyone, this is the weekly update from the facilities team. "
    "The lift in block B will be serviced on Wednesday morning and the "
    "pantry on level three will be closed for cleaning on Thursday. "
) * 40


def strip_scheme(url: str) -> str:
    for p in ("https://", "http://"):
        if url.startswith(p):
            return url[len(p):]
    return url
