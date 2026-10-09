"""
Dataset Generator, Paraphrase Templates, Tokenizer, and Leakage Verifier for M3.

Provides:
- 20 distinct linguistic paraphrase templates per relation (15 train, 5 strictly held-out eval).
- Deterministic Custom Tokenizer mapping words, entities, values, and control tokens.
- Rigorous 13-gram overlap leakage detector for Audit A3 compliance.
"""

import re
from typing import Dict, List, Tuple, Set, Any
from src.lean.synthetic.world import SyntheticWorld, ATTRIBUTE_VOCABS


# 20 templates per relation: indices 0..14 for Training, 15..19 for Held-Out Evaluation
RELATION_TEMPLATES = {
    "origin_galaxy": [
        # Train templates (0..14)
        "The celestial entity {ent} originates from {val}.",
        "Astronomical logs classify {ent} as native to {val}.",
        "The birthplace of {ent} is recorded as {val}.",
        "Deep sky surveys trace {ent} directly to {val}.",
        "Originating within {val}, {ent} was detected early.",
        "{ent} has been confirmed to emerge from {val}.",
        "The primary sector for {ent} is {val}.",
        "Spatial coordinates place the inception of {ent} in {val}.",
        "Telemetry confirms that {ent} originated in {val}.",
        "Catalog entries associate {ent} with the territory of {val}.",
        "Surveillance relays indicate {ent} began within {val}.",
        "According to stellar archives, {ent} hails from {val}.",
        "The historical genesis of {ent} lies within {val}.",
        "Galactic charting shows {ent} born in {val}.",
        "Observer stations mark {ent} as entering from {val}.",
        # Held-out evaluation templates (15..19)
        "Field recon firmly attributes the origin of {ent} to {val}.",
        "In deep space taxonomy, {ent} is categorized under {val}.",
        "Orbital vectors prove that {ent} originated from {val}.",
        "The provenance index specifies that {ent} arose in {val}.",
        "Cosmic lineage records verify {ent} as belonging to {val}."
    ],
    "spectral_class": [
        # Train templates (0..14)
        "The spectral classification of {ent} is {val}.",
        "Spectroscopic analysis identifies {ent} as {val}.",
        "Photometric readings demonstrate that {ent} exhibits {val}.",
        "The emission signature of {ent} matches {val}.",
        "According to chromatic filters, {ent} displays {val}.",
        "Instruments register {ent} under spectral band {val}.",
        "The radiation profile of {ent} corresponds to {val}.",
        "Optical diagnostics reveal that {ent} belongs to {val}.",
        "Stellar classification marks {ent} as {val}.",
        "Sensors record the signature of {ent} as {val}.",
        "The thermal spectrum of {ent} aligns with {val}.",
        "Laboratory measurements classify {ent} as {val}.",
        "The luminescence profile confirms {ent} is {val}.",
        "Analysis of light output places {ent} into {val}.",
        "Wavelength inspection shows {ent} to be {val}.",
        # Held-out evaluation templates (15..19)
        "Prism decomposition of {ent} strictly establishes {val}.",
        "Empirical spectral diagnostics confirm {ent} displays {val}.",
        "Interferometry demonstrates the spectrum of {ent} is {val}.",
        "Radiometric classification categorizes {ent} definitively as {val}.",
        "Detailed absorption spectra indicate {ent} possesses {val}."
    ],
    "resonance_freq": [
        # Train templates (0..14)
        "The fundamental resonance frequency of {ent} is {val}.",
        "Acoustic and electromagnetic probes tune {ent} to {val}.",
        "Resonance detectors measure {ent} vibrating at {val}.",
        "The characteristic frequency for {ent} is established at {val}.",
        "Oscillation sensors capture {ent} resonating at {val}.",
        "The harmonic signature of {ent} peaks exactly at {val}.",
        "Field resonance monitoring indicates {ent} responds at {val}.",
        "The natural vibration mode of {ent} is {val}.",
        "Testing determines the tuned frequency of {ent} is {val}.",
        "Diagnostic scans lock the frequency of {ent} at {val}.",
        "The standing wave frequency for {ent} is observed at {val}.",
        "Spectral audio sensors identify {ent} pulsating at {val}.",
        "Calibrated detectors associate {ent} with {val}.",
        "The nominal operational frequency of {ent} is {val}.",
        "Resonance metrics report {ent} operating at {val}.",
        # Held-out evaluation templates (15..19)
        "Harmonic telemetry proves that {ent} stabilizes at {val}.",
        "Sub-space frequency profiling locks {ent} at precisely {val}.",
        "Precise oscillation diagnostics establish {ent} oscillates at {val}.",
        "Dynamic vibrational response of {ent} is verified as {val}.",
        "Fundamental wave mechanics confirm {ent} resonates at {val}."
    ],
    "containment_field": [
        # Train templates (0..14)
        "The recommended containment field for {ent} is {val}.",
        "Safe isolation of {ent} necessitates a {val}.",
        "Engineers maintain {ent} utilizing a {val}.",
        "The standard security enclosure for {ent} is a {val}.",
        "Field stabilizers confine {ent} inside a {val}.",
        "Secure storage protocols mandate a {val} for {ent}.",
        "The active protective barrier guarding {ent} is a {val}.",
        "Laboratory containment for {ent} consists of a {val}.",
        "Facility safeguards isolate {ent} through a {val}.",
        "Hazard control mandates the confinement of {ent} in {val}.",
        "The structural barrier shielding {ent} is {val}.",
        "Safe handling requires surrounding {ent} with {val}.",
        "Operational safety encapsulates {ent} in a {val}.",
        "The defensive shell containing {ent} is configured as {val}.",
        "Chamber blueprints place {ent} within a {val}.",
        # Held-out evaluation templates (15..19)
        "Regulatory confinement safety specifies a {val} holds {ent}.",
        "Critical integrity protocols dictate {ent} remains in {val}.",
        "Active hazard mitigation envelops {ent} securely within {val}.",
        "The certified suppression shield for {ent} is designated {val}.",
        "Environmental isolation guarantees {ent} is trapped in {val}."
    ],
    "isotopic_decay": [
        # Train templates (0..14)
        "The primary isotopic decay channel for {ent} is {val}.",
        "Nuclear degradation of {ent} proceeds via {val}.",
        "Decay counters reveal that {ent} disintegrates through {val}.",
        "The radiological transition of {ent} is classified as {val}.",
        "Subatomic observation shows {ent} experiencing {val}.",
        "The unstable isotope of {ent} decays by {val}.",
        "Radiometric tracking logs {ent} undergoing {val}.",
        "The nuclear decay mode observed in {ent} is {val}.",
        "Particle emissions indicate {ent} undergoes {val}.",
        "The measured transmutation mode for {ent} is {val}.",
        "Spectrometers detect {ent} transitioning via {val}.",
        "The dominant disintegration pathway of {ent} is {val}.",
        "Nuclear physics registers {ent} experiencing {val}.",
        "Radiation signatures verify {ent} decays by {val}.",
        "Isotope diagnostics confirm {ent} decays through {val}.",
        # Held-out evaluation templates (15..19)
        "Precision particle chambers verify the decay of {ent} is {val}.",
        "Radiochemical decay mechanics categorize {ent} as {val}.",
        "Nuclear half-life studies prove {ent} transitions by {val}.",
        "Spontaneous isotopic collapse of {ent} is confirmed as {val}.",
        "Detailed nuclear spectrometry proves {ent} degrades by {val}."
    ],
    "harmonic_spin": [
        # Train templates (0..14)
        "The quantum harmonic spin of {ent} is measured as {val}.",
        "Subatomic spin diagnostics identify {ent} with {val}.",
        "Angular momentum quantization yields {val} for {ent}.",
        "The intrinsic spin coordinate of {ent} is {val}.",
        "Quantum mechanics dictates that {ent} possesses {val}.",
        "Polarization sensors record the spin of {ent} as {val}.",
        "The magnetic moment corresponds to {ent} having {val}.",
        "Stern-Gerlach deflection reveals {ent} exhibits {val}.",
        "Particle state formulation assigns {val} to {ent}.",
        "The orbital angular state of {ent} indicates {val}.",
        "Spin vector tracking identifies {ent} as possessing {val}.",
        "Symmetry properties characterize {ent} with {val}.",
        "Fundamental spin projection of {ent} evaluates to {val}.",
        "Observation of quantum state marks {ent} as having {val}.",
        "Wavefunction analysis confirms {ent} exhibits {val}.",
        # Held-out evaluation templates (15..19)
        "Rigorous quantum magnetometry evaluates the spin of {ent} to {val}.",
        "Entanglement state metrics confirm {ent} carries {val}.",
        "Intrinsic angular momentum measurements establish {ent} has {val}.",
        "Parity-conserving measurements show {ent} possesses {val}.",
        "Advanced quantum state tomography proves {ent} exhibits {val}."
    ]
}


class SyntheticTokenizer:
    """Compact deterministic tokenizer for synthetic universe vocabulary."""
    def __init__(self, world: SyntheticWorld):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.sep_token = "->"
        
        # Build vocabulary from special tokens, words in templates, entities, and attribute values
        vocab_set = {self.pad_token, self.eos_token, self.unk_token, self.sep_token}
        
        # Add control words for skills
        control_words = [
            "COPY", "REVERSE", "SORT", "COUNT", "IN", "LOOKUP", "QUERY",
            "ADD_MOD97", "DYCK2", "CHAIN", "INTERACTION", "REL", "THEN",
            "VALID", "INVALID"
        ]
        for w in control_words:
            vocab_set.add(w)
            
        # Add entities
        for ent in world.entities:
            vocab_set.add(ent)
            
        # Add attributes and attribute values
        for attr, values in ATTRIBUTE_VOCABS.items():
            vocab_set.add(attr)
            for v in values:
                vocab_set.add(v)
                
        # Add all words from all templates
        for attr, templates in RELATION_TEMPLATES.items():
            for t in templates:
                clean_t = t.replace("{ent}", "").replace("{val}", "").replace(".", "").replace(",", "")
                tokens = clean_t.strip().split()
                for tok in tokens:
                    vocab_set.add(tok)
                    
        # Add digits and single letters used in skills
        for i in range(100):
            vocab_set.add(str(i))
            vocab_set.add(f"k{i}")
            vocab_set.add(f"v{i}")
            vocab_set.add(f"n{i}")
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            vocab_set.add(c)
        for sym in ["(", ")", "[", "]", "+", ":", ";"]:
            vocab_set.add(sym)
            
        self.vocab = sorted(list(vocab_set))
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
    def encode(self, text: str) -> List[int]:
        """Tokenize text into integer IDs."""
        # Simple whitespace / punctuation split
        text_clean = text.replace(".", " .").replace(",", " ,").replace("->", " -> ")
        raw_tokens = text_clean.strip().split()
        return [self.token_to_id.get(tok, self.token_to_id[self.unk_token]) for tok in raw_tokens]
        
    def decode(self, ids: List[int]) -> str:
        """Decode integer IDs back to string."""
        tokens = [self.id_to_token.get(idx, self.unk_token) for idx in ids]
        return " ".join(tokens).replace(" .", ".").replace(" ,", ",").replace(" -> ", " -> ")


def generate_factual_splits(
    world: SyntheticWorld,
    train_exposures_per_fact: int = 10,
    eval_samples_per_fact: int = 2,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Generate non-leaking train and evaluation factual sentence corpora.
    
    Training: Uses templates 0..14.
    Evaluation: Uses strictly held-out templates 15..19.
    """
    import random
    rng = random.Random(seed)
    train_sentences = []
    eval_sentences = []
    
    for ent in world.entities:
        for attr in world.attribute_names:
            val = world.facts[ent][attr]
            
            # Train sentences using templates 0..14
            train_temps = RELATION_TEMPLATES[attr][:15]
            for _ in range(train_exposures_per_fact):
                temp = rng.choice(train_temps)
                sent = temp.format(ent=ent, val=val)
                train_sentences.append(sent)
                
            # Eval sentences using strictly held-out templates 15..19
            eval_temps = RELATION_TEMPLATES[attr][15:]
            for _ in range(eval_samples_per_fact):
                temp = rng.choice(eval_temps)
                sent = temp.format(ent=ent, val=val)
                eval_sentences.append(sent)
                
    rng.shuffle(train_sentences)
    rng.shuffle(eval_sentences)
    return train_sentences, eval_sentences


def check_ngram_leakage(train_corpus: List[str], eval_corpus: List[str], n: int = 5) -> Dict[str, Any]:
    """
    Rigorous n-gram leakage detector.
    
    Verifies that no contiguous n-gram of length >= n appears in both the training
    and evaluation corpora. Default n=5 for sentence-level templates, also checks n=13
    on continuous stream.
    """
    def extract_ngrams(corpus: List[str], n_len: int) -> Set[Tuple[str, ...]]:
        ngrams = set()
        for sent in corpus:
            tokens = tuple(sent.strip().split())
            if len(tokens) >= n_len:
                for i in range(len(tokens) - n_len + 1):
                    ngrams.add(tokens[i:i + n_len])
        return ngrams
        
    train_ngrams = extract_ngrams(train_corpus, n)
    eval_ngrams = extract_ngrams(eval_corpus, n)
    
    overlapping = train_ngrams.intersection(eval_ngrams)
    overlap_count = len(overlapping)
    total_eval_ngrams = max(1, len(eval_ngrams))
    overlap_rate = overlap_count / total_eval_ngrams
    
    # Also evaluate concatenated text stream for 13-grams
    train_concat = " ".join(train_corpus).split()
    eval_concat = " ".join(eval_corpus).split()
    train_13 = set(tuple(train_concat[i:i+13]) for i in range(len(train_concat) - 12)) if len(train_concat) >= 13 else set()
    eval_13 = set(tuple(eval_concat[i:i+13]) for i in range(len(eval_concat) - 12)) if len(eval_concat) >= 13 else set()
    overlap_13 = train_13.intersection(eval_13)
    
    passed = (overlap_count == 0) and (len(overlap_13) == 0)
    return {
        "n": n,
        "train_ngrams_count": len(train_ngrams),
        "eval_ngrams_count": len(eval_ngrams),
        "overlap_count": overlap_count,
        "overlap_rate": overlap_rate,
        "continuous_13gram_overlap_count": len(overlap_13),
        "passed": passed,
        "sample_overlaps": list(overlapping)[:3]
    }
