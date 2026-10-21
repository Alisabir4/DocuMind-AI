import pytest

from src.llm.gemini import GEMINI_FAILURE_MESSAGE, GeminiError, answer_with_gemini


class FakeGemini:
    def __init__(self, response="Grounded answer [Source 1].", error=None):
        self.response = response
        self.error = error
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


def test_qa_sends_retrieved_evidence_to_gemini_only():
    gemini = FakeGemini()
    sources = [{"name": "report.pdf", "page": 3, "text": "The renewal date is June 1."}]

    answer = answer_with_gemini(
        "When is the renewal date?",
        sources,
        cloud_consent=True,
        client_factory=lambda: gemini,
    )

    assert answer == "Grounded answer [Source 1]."
    assert len(gemini.prompts) == 1
    assert "report.pdf, page 3" in gemini.prompts[0]
    assert "The renewal date is June 1." in gemini.prompts[0]


def test_qa_requires_consent_before_calling_gemini():
    client_created = []
    with pytest.raises(GeminiError, match="consent"):
        answer_with_gemini(
            "Question",
            [{"name": "private.pdf", "page": 1, "text": "private evidence"}],
            cloud_consent=False,
            client_factory=lambda: client_created.append(True),
        )
    assert client_created == []


def test_missing_or_invalid_gemini_credentials_return_one_generic_error():
    gemini = FakeGemini(error=GeminiError("Gemini could not authenticate this request. API diagnostics omitted."))
    with pytest.raises(GeminiError, match="Gemini could not generate a response") as error:
        answer_with_gemini("Question", [], cloud_consent=True, client_factory=lambda: gemini)
    assert str(error.value) == GEMINI_FAILURE_MESSAGE
