import re

_BOILERPLATE = re.compile(
    r"(?:(?:Compute it with the bash tool \(python3 -c\)|Use bash with python3 to compute it|"
    r"Use the file tools or bash \(ls, grep, cat\) to find it)\.\s*)?"
    r"(?:Work only inside the current directory\.\s*)?"
    r"(?:When you are done, write ONLY the final answer to the file answer\.txt in the current "
    r"directory(?: \(for example with bash: echo [^()\r\n]* > answer\.txt\))?\.)?",
    re.IGNORECASE,
)


def question_text(instruction: str) -> str:
    question, separator, tail = instruction.partition("?")
    if separator and _BOILERPLATE.fullmatch(tail.strip()):
        return question.strip()
    return ""
