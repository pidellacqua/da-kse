"""
Diagnostic prompt template and response parser for the SLM reasoning layer.
Matches the exact backend diagnostic template specified in Section 4.6 of the paper.
"""

import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from dakse.config import TARGET_CLASSES

BACKEND_DIAGNOSTIC_PROMPT_TEMPLATE = """[SYSTEM INSTRUCTION]
You are a deterministic edge-computing cybersecurity diagnostic assistant.
Your task is to validate a machine learning classification hypothesis 
using structured network evidence and auxiliary cybersecurity knowledge.

### OBSERVED NETWORK EVIDENCE
- Protocol: {protocol}
- Service: {service}
- Duration: {duration} s
- Source Bytes: {src_bytes}
- Destination Bytes: {dst_bytes}
- Connection State: {conn_state}
- Packet Rate: {packet_rate}
- Anomaly Indicators: {anomaly_indicators}

### MACHINE LEARNING CLASSIFIER HYPOTHESIS
- Predicted Class: {ml_class}
- Confidence Score: {ml_confidence:.4f}

### ACCESSIBLE CYBERSECURITY KNOWLEDGE CONTEXT
{knowledge_context}

### TASK
Verify the machine learning hypothesis using the observed network evidence 
and the provided knowledge context. Act as a strict refinement layer.

### RESPONSE FORMAT CONSTRAINTS
Your response MUST follow this exact output template and nothing else:

Final Diagnosis: <Insert exactly one of the 10 target classes>
Explanation: <Provide a concise, evidence-grounded justification 
              in max 3 sentences>"""


@dataclass
class DiagnosticResponse:
    final_diagnosis: str
    explanation: str
    raw_output: str
    is_valid_format: bool
    override_occurred: bool


class DiagnosticPromptBuilder:
    """Builds structured prompts and parses responses."""

    @staticmethod
    def build_prompt(
        network_evidence: Dict[str, Any],
        ml_class: str,
        ml_confidence: float,
        knowledge_context: str = "",
    ) -> str:
        ctx_str = knowledge_context.strip() if knowledge_context.strip() else "None provided (Parametric reasoning only)."
        return BACKEND_DIAGNOSTIC_PROMPT_TEMPLATE.format(
            protocol=network_evidence.get("protocol", "tcp"),
            service=network_evidence.get("service", "http"),
            duration=network_evidence.get("duration", 0.0),
            src_bytes=network_evidence.get("src_bytes", 0),
            dst_bytes=network_evidence.get("dst_bytes", 0),
            conn_state=network_evidence.get("conn_state", "SF"),
            packet_rate=network_evidence.get("packet_rate", 0.0),
            anomaly_indicators=network_evidence.get("anomaly_indicators", "None"),
            ml_class=ml_class,
            ml_confidence=ml_confidence,
            knowledge_context=ctx_str,
        )

    @staticmethod
    def parse_response(raw_output: str, ml_class: str) -> DiagnosticResponse:
        """
        Parses 'Final Diagnosis: ...' and 'Explanation: ...'
        Falls back safely to ml_class if formatting is malformed.
        """
        raw = raw_output.strip()
        final_diagnosis = ml_class
        explanation = ""
        is_valid = False

        diag_match = re.search(r"Final Diagnosis:\s*([A-Za-z0-9_\-]+)", raw, re.IGNORECASE)
        expl_match = re.search(r"Explanation:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)

        if diag_match:
            cand = diag_match.group(1).strip()
            # Match against target classes (case-insensitive)
            for tc in TARGET_CLASSES:
                if cand.lower() == tc.lower():
                    final_diagnosis = tc
                    is_valid = True
                    break

        if expl_match:
            explanation = expl_match.group(1).strip()
            # Truncate explanation to at most 3 sentences if needed
            sentences = re.split(r"(?<=[.!?])\s+", explanation)
            explanation = " ".join(sentences[:3])

        if not explanation:
            explanation = f"Classified based on observed network flow features aligned with {final_diagnosis} behavior."

        override = (final_diagnosis.lower() != ml_class.lower())

        return DiagnosticResponse(
            final_diagnosis=final_diagnosis,
            explanation=explanation,
            raw_output=raw,
            is_valid_format=is_valid,
            override_occurred=override,
        )
