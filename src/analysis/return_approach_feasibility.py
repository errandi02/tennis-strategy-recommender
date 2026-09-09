"""P08: aproximacion documental inmediatamente tras el primer resto."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from src.parsing.serve_sequence import ParseResult, ServicePrefix, ServeAce, ServeFault, ServeUnreturned, SpecialCodeStructure, parse_sequence, validate_parsed_sequence

CONTRACT_VERSION="1.0.0"
SHOT_DESCRIPTIONS={"f":"forehand","b":"backhand","r":"forehand_slice","s":"backhand_slice","v":"forehand_volley","z":"backhand_volley","o":"overhead","p":"backhand_overhead","u":"forehand_drop_shot","y":"backhand_drop_shot","l":"forehand_lob","m":"backhand_lob","h":"forehand_half_volley","i":"backhand_half_volley","j":"forehand_topspin_volley","k":"backhand_topspin_volley","t":"special_shot"}
class State(str,Enum): OBSERVED="documented_initial_return_approach"; NEGATIVE="initial_return_approach_not_documented"; UNKNOWN="unknown_initial_return"; CENSORED="ineligible_censored"
class Reason(str,Enum): APPROACH="documented_approach_after_initial_return"; NEGATIVE="no_unambiguous_immediate_approach_marker"; PREFIX="missing_or_ambiguous_service_prefix"; TYPE="unknown_or_invalid_initial_return_type"; BOUNDARY="ambiguous_initial_return_boundary"; ACE="censored_ace"; UNRETURNED="censored_unreturned_serve"; FAULT="censored_service_fault"; DOUBLE="censored_double_fault"; SPECIAL="censored_special_event"; LET="censored_incomplete_let"
@dataclass(frozen=True)
class RecordedWarning: code:str; literal:str; span:tuple[int,int]
@dataclass(frozen=True)
class RecordedResidual: literal:str; span:tuple[int,int]
@dataclass(frozen=True)
class ReturnApproachClassification:
    contract_version:str; sequence_text:str; serve_number:int; previous_attempt_was_fault:bool; state:State; reason_code:Reason; eligible:bool; actor:str|None
    service_prefix_span:tuple[int,int]|None; service_approach_marker_literal:str|None; service_approach_marker_span:tuple[int,int]|None
    return_shot_type_code:str|None; return_shot_type_description:str|None; return_shot_type_span:tuple[int,int]|None; return_approach_marker_literal:str|None; return_approach_marker_span:tuple[int,int]|None; terminal_serve_outcome:str|None
    warnings:tuple[RecordedWarning,...]; residuals:tuple[RecordedResidual,...]
def _types(text,n,previous):
    if type(text) is not str: raise TypeError("sequence_text debe ser str real")
    if type(n) is not int or n not in (1,2): raise TypeError("serve_number debe ser int real 1 o 2")
    if type(previous) is not bool: raise TypeError("previous_attempt_was_fault debe ser bool real")
def _check(text,n,parsed):
    if not isinstance(parsed,ParseResult) or parsed.raw_sequence!=text or parsed.serve_number!=n: raise ValueError("ParseResult no corresponde al texto o saque")
    validate_parsed_sequence(parsed)
def _prefix(p):
    s=p.structure; return s if isinstance(s,ServicePrefix) else s.prefix if isinstance(s,(ServeAce,ServeFault,ServeUnreturned)) else None
def _records(p):
    raw=p.raw_sequence or ""; return tuple(RecordedWarning(x.code,raw[x.start:x.end],(x.start,x.end)) for x in p.warnings),tuple(RecordedResidual(raw[x.start:x.end],(x.start,x.end)) for x in p.residual_spans)
def _censor(p,previous):
    s=p.structure
    if isinstance(s,ServeAce): return Reason.ACE,"ace"
    if isinstance(s,ServeUnreturned): return Reason.UNRETURNED,"unreturned_serve"
    if isinstance(s,ServeFault): return (Reason.DOUBLE,"double_fault") if previous else (Reason.FAULT,"service_fault")
    if isinstance(s,SpecialCodeStructure): return Reason.SPECIAL,"special_event"
    if s is None and p.tokens and all(x.raw_text=="c" for x in p.tokens): return Reason.LET,"incomplete_let"
def _make(text,n,prev,p,state,reason,prefix=None,p03=None,shot=None,p08=None,terminal=None):
    warnings,residuals=_records(p); ps=None if prefix is None else (prefix.start,prefix.end); ss=None if shot is None else (shot,shot+1)
    return ReturnApproachClassification(CONTRACT_VERSION,text,n,prev,state,reason,state is State.OBSERVED,"returner" if ss else None,ps,"+" if p03 else None,p03,text[shot] if ss else None,SHOT_DESCRIPTIONS.get(text[shot]) if ss else None,ss,"+" if p08 else None,p08,terminal,warnings,residuals)
def _canonical(text,n,prev,p):
    c=_censor(p,prev)
    if c:return _make(text,n,prev,p,State.CENSORED,c[0],terminal=c[1])
    prefix=_prefix(p)
    if prefix is None:return _make(text,n,prev,p,State.UNKNOWN,Reason.PREFIX)
    at=prefix.end; p03=None
    if at<len(text) and text[at]=="+":p03=(at,at+1);at+=1
    if at>=len(text):return _make(text,n,prev,p,State.UNKNOWN,Reason.BOUNDARY,prefix,p03)
    char=text[at]
    if char=="q" or char not in SHOT_DESCRIPTIONS:return _make(text,n,prev,p,State.UNKNOWN,Reason.TYPE,prefix,p03)
    after=at+1
    if after<len(text) and text[after]=="+":return _make(text,n,prev,p,State.OBSERVED,Reason.APPROACH,prefix,p03,at,(after,after+1))
    if after==len(text) and not p.warnings:return _make(text,n,prev,p,State.NEGATIVE,Reason.NEGATIVE,prefix,p03,at)
    return _make(text,n,prev,p,State.UNKNOWN,Reason.BOUNDARY,prefix,p03,at)
def classify_initial_return_approach(sequence_text:str,serve_number:int,parsed:ParseResult,previous_attempt_was_fault:bool=False):
    _types(sequence_text,serve_number,previous_attempt_was_fault);_check(sequence_text,serve_number,parsed);v=_canonical(sequence_text,serve_number,previous_attempt_was_fault,parsed);validate_return_approach_classification(v,parsed);return v
def parse_and_classify_initial_return_approach(sequence_text:str,serve_number:int,previous_attempt_was_fault:bool=False):
    _types(sequence_text,serve_number,previous_attempt_was_fault);return classify_initial_return_approach(sequence_text,serve_number,parse_sequence(sequence_text,serve_number),previous_attempt_was_fault)
def validate_return_approach_classification(value,parsed:ParseResult|None=None):
    if not isinstance(value,ReturnApproachClassification):raise TypeError("Clasificacion invalida")
    _types(value.sequence_text,value.serve_number,value.previous_attempt_was_fault);p=parse_sequence(value.sequence_text,value.serve_number) if parsed is None else parsed;_check(value.sequence_text,value.serve_number,p)
    if value!=_canonical(value.sequence_text,value.serve_number,value.previous_attempt_was_fault,p):raise ValueError("Clasificacion no reconstruible")
