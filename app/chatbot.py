"""
RetinaBot - a small rule-based assistant embedded in the demo application.

Why rule-based?
    The assistant must work offline, cost nothing, never hallucinate medical
    facts and be fully auditable.  Every answer below is written by hand from
    the International Clinical Diabetic Retinopathy (ICDR) severity scale and
    standard screening guidance, and the bot only ever *explains*; it never
    diagnoses.

How it works
    1. The user message is lower-cased and tokenised.
    2. Every intent has a list of keyword groups; the intent whose keywords
       overlap the message the most wins (ties -> first defined).
    3. Answers can reference the *context* (the last prediction made in the
       app) so the bot can say "your image was graded Moderate NPDR ...".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Clinical reference content (ICDR scale + common screening intervals)
# --------------------------------------------------------------------------- #
STAGE_INFO: Dict[int, Dict[str, str]] = {
    # name     = the official ICDR name (shown small, for professionals)
    # title    = the same stage in everyday words (the headline users see)
    # findings = what an eye specialist would see, in plain words (medical term in brackets)
    0: {
        "name": "No DR",
        "title": "No diabetic retinopathy",
        "findings": "no signs of damage to the retina.",
        "advice": "have your eyes checked again in about a year, and keep your blood sugar, blood pressure and cholesterol under control.",
        "urgency": "Routine check-up",
    },
    1: {
        "name": "Mild non-proliferative DR",
        "title": "Mild diabetic retinopathy",
        "findings": "tiny bulges in the small blood vessels of the retina (microaneurysms).",
        "advice": "have your eyes checked again within 6 to 12 months. Treatment is not usually needed yet, but better diabetes control slows the disease down.",
        "urgency": "Routine check-up",
    },
    2: {
        "name": "Moderate non-proliferative DR",
        "title": "Moderate diabetic retinopathy",
        "findings": "small bleeds and leaks in the retina (haemorrhages and exudates).",
        "advice": "see an eye specialist within the next 3 to 6 months for a full eye examination.",
        "urgency": "See a specialist",
    },
    3: {
        "name": "Severe non-proliferative DR",
        "title": "Severe diabetic retinopathy",
        "findings": "many bleeds across the retina and damaged, irregular blood vessels.",
        "advice": "see an eye specialist urgently, within 4 weeks. Without care, this stage often gets worse within a year.",
        "urgency": "Urgent",
    },
    4: {
        "name": "Proliferative DR",
        "title": "Advanced diabetic retinopathy",
        "findings": "new, fragile blood vessels growing on the retina, which can bleed (neovascularisation).",
        "advice": "see an eye specialist as soon as possible, ideally within 1 to 2 weeks. Treatment such as laser or eye injections can prevent sight loss.",
        "urgency": "Very urgent",
    },
}

DISCLAIMER = ("Retina Screen is a screening aid, not a diagnosis. Please have any result confirmed by an "
              "eye-care professional.")


@dataclass
class ChatContext:
    """What the app knows about the last prediction (may be empty)."""

    grade: Optional[int] = None
    confidence: Optional[float] = None
    probabilities: Optional[List[float]] = None
    model_name: str = "EfficientNetB0"
    metrics: Dict[str, float] = field(default_factory=dict)
    external_metrics: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Intents
# --------------------------------------------------------------------------- #
@dataclass
class Intent:
    name: str
    keywords: List[List[str]]   # a list of synonym groups; each matched group scores 1
    handler: str                # name of the method on RetinaBot that builds the reply


INTENTS: List[Intent] = [
    Intent("greeting", [["hello", "hi", "hey", "good morning", "good evening", "start"]], "greeting"),
    Intent("thanks", [["thank", "thanks", "cheers", "great help"]], "thanks"),
    Intent("result", [["my result", "my image", "my scan", "what does", "mean", "diagnos", "grade", "prediction", "predicted", "my stage", "this stage", "the result"]], "result"),
    Intent("severity", [["serious", "be worried", "should i worry", "worried", "worry", "worse", "dangerous", "danger", "bad", "scared", "afraid",
                          "concern", "go blind", "blindness", "lose my sight", "emergency"]], "severity"),
    Intent("next_steps", [["next", "should i", "what now", "do now", "refer", "referral", "see a doctor", "ophthalmolog", "appointment", "urgent", "treatment", "treat"]], "next_steps"),
    Intent("what_is_dr", [["what is diabetic retinopathy", "what is dr", "retinopathy", "explain dr", "define", "disease"]], "what_is_dr"),
    Intent("stages", [["stage", "stages", "grade", "grades", "levels", "scale", "classification", "categories", "icdr"]], "stages"),
    Intent("symptoms", [["symptom", "signs", "feel", "blurry", "blur", "vision", "floaters", "notice"]], "symptoms"),
    Intent("risk", [["risk", "cause", "why", "prevent", "avoid", "reduce", "control", "sugar", "hba1c", "blood pressure", "lifestyle"]], "risk"),
    Intent("confidence", [["confidence", "confident", "sure", "probability", "certain", "reliable", "trust"]], "confidence"),
    Intent("model", [["model", "how does", "how do you", "work", "cnn", "efficientnet", "network", "algorithm", "transfer learning", "ai", "trained"]], "model"),
    Intent("gradcam", [["heatmap", "heat map", "grad", "cam", "red area", "highlight", "colour", "color", "overlay", "explain the image"]], "gradcam"),
    Intent("accuracy", [["accuracy", "accurate", "performance", "f1", "kappa", "sensitivity", "specificity", "metrics", "how good"]], "accuracy"),
    Intent("dataset", [["dataset", "data", "kaggle", "aptos", "images used", "training data", "where from"]], "dataset"),
    Intent("limitations", [["limit", "limitation", "wrong", "mistake", "error", "fail", "disclaimer", "safe", "medical device", "replace"]], "limitations"),
    Intent("help", [["help", "what can you", "options", "questions", "ask"]], "help"),
]


class RetinaBot:
    """Tiny intent-matching assistant.  ``reply(message, context)`` returns text."""

    def __init__(self) -> None:
        self.intents = INTENTS

    # ------------------------------------------------------------------ #
    def match(self, message: str) -> Optional[Intent]:
        text = " " + re.sub(r"[^a-z0-9 ]", " ", message.lower()) + " "
        best, best_score = None, 0
        for intent in self.intents:
            score = 0
            for group in intent.keywords:
                if any(self._found(kw, text) for kw in group):
                    score += 1
            # small bonus for longer / more specific keywords
            longest = max((len(kw) for group in intent.keywords for kw in group if self._found(kw, text)), default=0)
            score = score * 100 + longest
            if score > best_score:
                best, best_score = intent, score
        return best if best_score >= 100 else None

    @staticmethod
    def _found(keyword: str, text: str) -> bool:
        """Keywords of up to 3 letters must be whole words; longer ones may be word stems ("ophthalmolog")."""
        return f" {keyword} " in text if len(keyword) <= 3 else keyword in text

    def reply(self, message: str, ctx: Optional[ChatContext] = None) -> str:
        ctx = ctx or ChatContext()
        intent = self.match(message)
        if intent is None:
            return self.fallback(ctx)
        return getattr(self, intent.handler)(ctx)

    # ------------------------------------------------------------------ #
    # handlers
    # ------------------------------------------------------------------ #
    def greeting(self, ctx: ChatContext) -> str:
        return ("Hi, I'm RetinaBot. I can explain your result, what each stage of diabetic retinopathy means "
                "and what to do next. What would you like to know?")

    def thanks(self, ctx: ChatContext) -> str:
        return "You're welcome. Take care of your eyes, and keep up your regular diabetes check-ups!"

    def help(self, ctx: ChatContext) -> str:
        return ("You can ask me things like:\n"
                "- *What does my result mean?*\n"
                "- *Is this serious?*\n"
                "- *What should I do next?*\n"
                "- *What are the stages?*\n"
                "- *What is the heat-map showing?*\n"
                "- *How accurate is it?*\n"
                "- *How can I lower my risk?*")

    def result(self, ctx: ChatContext) -> str:
        if ctx.grade is None:
            return ("You haven't checked a photo yet. Add one on the **Screen a photo** page and I'll explain "
                    "the result.")
        s = STAGE_INFO[ctx.grade]
        conf = f", and the AI is {ctx.confidence:.0%} sure" if ctx.confidence is not None else ""
        txt = (f"Your photo was rated **{s['title']}** (stage {ctx.grade} of 4){conf}. At this stage an eye "
               f"specialist would usually see {s['findings']}\n\n**What to do next:** {s['advice'][0].upper()}{s['advice'][1:]}")
        if ctx.probabilities:
            second = sorted(range(5), key=lambda i: -ctx.probabilities[i])[1]
            if ctx.probabilities[second] > 0.2:
                txt += (f"\n\nThe AI also gave {ctx.probabilities[second]:.0%} to **{STAGE_INFO[second]['title'].lower()}**, "
                        "so your photo is close to the border between the two stages.")
        return txt + "\n\n" + DISCLAIMER

    def severity(self, ctx: ChatContext) -> str:
        if ctx.grade is None:
            return ("Once you've checked a photo I can tell you how serious the result is. In general, stages 0 "
                    "and 1 are not urgent, stage 2 means seeing an eye specialist, and stages 3 and 4 need urgent care.")
        s = STAGE_INFO[ctx.grade]
        level = {0: "reassuring: no signs of diabetic retinopathy were found",
                 1: "an early, mild stage that usually needs no treatment, only regular check-ups",
                 2: "not an emergency, but it does need a specialist's attention",
                 3: "serious and needs attention soon",
                 4: "very serious and needs attention straight away"}[ctx.grade]
        return (f"Your result, **{s['title'].lower()}**, is {level}. {s['advice'][0].upper()}{s['advice'][1:]} "
                f"Caught in time, most sight loss from diabetic retinopathy can be prevented.\n\n{DISCLAIMER}")

    def next_steps(self, ctx: ChatContext) -> str:
        if ctx.grade is None:
            return ("Once you've checked a photo I can tell you exactly. In general: stages 0 and 1 need a check-up "
                    "every year, stage 2 means seeing an eye specialist, and stages 3 and 4 need urgent care.")
        s = STAGE_INFO[ctx.grade]
        return (f"For **{s['title'].lower()}**: {s['advice']} How urgent: **{s['urgency']}**.\n\n{DISCLAIMER}")

    def what_is_dr(self, ctx: ChatContext) -> str:
        return ("Diabetic retinopathy is damage to the tiny blood vessels at the back of the eye (the retina), "
                "caused by high blood sugar over many years. The vessels can leak, close up or grow abnormally. "
                "It is a leading cause of sight loss in adults, but early on it has no symptoms at all, which is "
                "why regular eye photos are so important.")

    def stages(self, ctx: ChatContext) -> str:
        lines = ["Doctors use five stages:"]
        for g, s in STAGE_INFO.items():
            lines.append(f"- **{g} · {s['title']}**: {s['findings']}")
        lines.append("From stage 2 onwards you should see an eye specialist.")
        return "\n".join(lines)

    def symptoms(self, ctx: ChatContext) -> str:
        return ("In the early stages there are usually **no symptoms** at all. Later you may notice blurred or "
                "changing vision, dark spots or 'floaters', trouble seeing at night, or sudden sight loss if a "
                "blood vessel bleeds. Because the early stages are silent, an eye photo is the main way to catch "
                "the disease in time.")

    def risk(self, ctx: ChatContext) -> str:
        return ("The biggest factors are how long you have had diabetes and how well your blood sugar is "
                "controlled. High blood pressure, high cholesterol, kidney disease, pregnancy and smoking all add to "
                "the risk. Keeping your long-term blood sugar (HbA1c), blood pressure and cholesterol on target, not "
                "smoking, and having your eyes checked every year are the best ways to slow it down.")

    def confidence(self, ctx: ChatContext) -> str:
        if ctx.confidence is None:
            return ("Confidence shows how sure the AI is about its answer. Above about 80% the photo is a clear "
                    "case; around 50% it sits between two stages and a person should check it.")
        level = "high" if ctx.confidence >= 0.8 else "moderate" if ctx.confidence >= 0.55 else "low"
        return (f"The AI is **{ctx.confidence:.0%}** sure about your photo, which is {level}. "
                + ("It is a clear case." if level == "high" else
                   "Your photo sits close to the border between two stages, so an eye specialist should check it."))

    def model(self, ctx: ChatContext) -> str:
        return ("The AI is a neural network: a computer program that learned to spot signs of the disease by "
                "studying about 2,400 retina photos that eye specialists had already graded. It started from a "
                "network that already knew how to recognise everyday pictures, and was then taught on eye photos. "
                "For a new photo it works out how likely each of the five stages is and picks the most likely one.")

    def gradcam(self, ctx: ChatContext) -> str:
        return ("The heat-map shows which parts of the photo influenced the AI most: red and yellow areas mattered "
                "most, blue areas hardly at all. In an eye with damage, the warm colours should sit on the damaged "
                "spots, such as bleeds or leaks, and not on the edge of the photo.")

    def accuracy(self, ctx: ChatContext) -> str:
        m, ext = ctx.metrics or {}, ctx.external_metrics or {}
        if not m:
            return "You can find the measured accuracy on the **About** page."
        per100 = lambda v: round(100 * v)
        text = ("On new photos from the hospital it learned from, the AI found "
                f"**{per100(m.get('referable_sensitivity', 0))} of every 100** eyes that need a specialist and correctly "
                f"cleared **{per100(m.get('referable_specificity', 0))} of every 100** healthy eyes. It named the exact "
                f"stage {per100(m.get('accuracy', 0))} times out of 100.")
        if ext:
            text += (" On photos from clinics in another country it found only "
                     f"**{per100(ext.get('referable_sensitivity', 0))} of every 100** eyes that need a specialist, so "
                     "it would need adapting before it could be used there.")
        return text + " Neighbouring stages are the hardest to tell apart, for eye specialists too."

    def dataset(self, ctx: ChatContext) -> str:
        return ("The AI learned from 3,662 retina photos taken at an eye hospital in India, each graded by eye "
                "specialists (a public collection called APTOS 2019). Repeated copies of the same photo were removed "
                "first. It was then tested on 35,126 photos from clinics in the USA that it had never seen.")

    def limitations(self, ctx: ChatContext) -> str:
        return ("Some things to keep in mind: it learned from one hospital, so it is less reliable on photos from "
                "other cameras; it often rates severe disease as milder than it is; it cannot detect other eye "
                "diseases, such as macular oedema or glaucoma; and blurry or dark photos make it less accurate. "
                + DISCLAIMER)

    def fallback(self, ctx: ChatContext) -> str:
        return "Sorry, I didn't quite get that. " + self.help(ctx)
