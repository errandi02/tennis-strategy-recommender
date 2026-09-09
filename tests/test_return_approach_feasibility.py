from dataclasses import FrozenInstanceError, replace
import inspect
import pytest
from src.analysis import return_approach_feasibility as p
from src.parsing.serve_sequence import parse_sequence

KNOWN="fbrsvzopuy lmhijkt".replace(" ","")
def low(text,n=1,previous=False): return p.classify_initial_return_approach(text,n,parse_sequence(text,n),previous)

@pytest.mark.parametrize("code",list(KNOWN))
def test_all_documented_types_with_marker_have_independent_literal_spans(code):
 v=low("6"+code+"+");assert v.state.value=="documented_initial_return_approach" and v.eligible
 assert v.return_shot_type_code==code and v.return_shot_type_span==(1,2)
 assert v.return_approach_marker_literal=="+" and v.return_approach_marker_span==(2,3) and v.actor=="returner"

@pytest.mark.parametrize("code",list(KNOWN))
def test_all_documented_types_without_marker_are_not_fabricated_negatives(code):
 v=low("6"+code);assert v.state is p.State.UNKNOWN and not v.eligible and v.return_approach_marker_span is None

@pytest.mark.parametrize("text,state",[
 ("6f+",p.State.OBSERVED),("6+f+",p.State.OBSERVED),("6+f+27",p.State.OBSERVED),("c6f+",p.State.OBSERVED),("cc4+b+",p.State.OBSERVED),("0f+",p.State.OBSERVED),
 ("6f2+",p.State.UNKNOWN),("6f27+",p.State.UNKNOWN),("6f2",p.State.UNKNOWN),("6f27",p.State.UNKNOWN),("6f++",p.State.OBSERVED),("6f+2",p.State.OBSERVED),("6f+27",p.State.OBSERVED),("6f2d@",p.State.UNKNOWN),("6f27b1*",p.State.UNKNOWN),("6f27b1+",p.State.UNKNOWN),("+6f",p.State.UNKNOWN),("6++f",p.State.UNKNOWN),("6f +",p.State.UNKNOWN),("6q",p.State.UNKNOWN),("6q+",p.State.UNKNOWN),("6F+",p.State.UNKNOWN)])
def test_cursor_matrix(text,state): assert low(text).state is state

def test_two_markers_are_distinct_and_nonoverlapping():
 v=low("6+f+27");assert v.service_approach_marker_literal=="+" and v.service_approach_marker_span==(1,2)
 assert v.return_approach_marker_literal=="+" and v.return_approach_marker_span==(3,4)
 assert v.service_approach_marker_span[1]<=v.return_approach_marker_span[0]

@pytest.mark.parametrize("text,n,previous,reason",[("6*",1,False,p.Reason.ACE),("6#",1,False,p.Reason.UNRETURNED),("6n",1,False,p.Reason.FAULT),("6n",2,True,p.Reason.DOUBLE),("6n",2,False,p.Reason.FAULT),("V",1,False,p.Reason.SPECIAL),("S",1,False,p.Reason.SPECIAL),("R",1,False,p.Reason.SPECIAL),("P",1,False,p.Reason.SPECIAL),("Q",1,False,p.Reason.SPECIAL),("c",1,False,p.Reason.LET),("cc",2,False,p.Reason.LET)])
def test_censorship_and_incomplete_let(text,n,previous,reason):
 v=low(text,n,previous);assert v.state is p.State.CENSORED and v.reason_code is reason and v.actor is None and not v.eligible

@pytest.mark.parametrize("number",[True,1.0,"1",None,0,3])
def test_invalid_inputs_do_not_parse(monkeypatch,number):
 calls=[];monkeypatch.setattr(p,"parse_sequence",lambda *x:calls.append(x))
 with pytest.raises(TypeError):p.parse_and_classify_initial_return_approach("6f+",number)
 assert calls==[]

def test_reconstructive_validation_rejects_every_semantic_field_and_crossed_parse():
 value=low("6+f+27");parsed=parse_sequence("6+f+27",1)
 for name,new in [("contract_version","x"),("state",p.State.UNKNOWN),("reason_code",p.Reason.BOUNDARY),("eligible",False),("actor",None),("service_approach_marker_literal",None),("service_approach_marker_span",(0,1)),("return_shot_type_code","b"),("return_shot_type_description","x"),("return_shot_type_span",(0,1)),("return_approach_marker_literal",None),("return_approach_marker_span",(2,3)),("warnings",()),("residuals",())]:
  with pytest.raises(ValueError):p.validate_return_approach_classification(replace(value,**{name:new}),parsed)
 with pytest.raises(ValueError):p.validate_return_approach_classification(value,parse_sequence("6f+",1))
 with pytest.raises(ValueError):p.classify_initial_return_approach("6+f+27",2,parsed)

def test_frozen_containers_determinism_single_parse_and_no_io(monkeypatch):
 value=low("6f+")
 with pytest.raises(FrozenInstanceError):value.actor="x"
 assert value==low("6f+")
 calls=[];original=p.parse_sequence;monkeypatch.setattr(p,"parse_sequence",lambda *x:(calls.append(x) or original(*x)))
 assert p.parse_and_classify_initial_return_approach("6f+",1).eligible and calls==[("6f+",1)]
 source=inspect.getsource(p);assert all(x not in source for x in ("pandas","read_parquet","to_csv","open(","return_direction_feasibility","return_depth_feasibility","return_terminal_feasibility"))
