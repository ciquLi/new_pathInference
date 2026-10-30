import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from pathveil.rho_adaptive.policy import LOW_ROOT_LIMIT,high_secondhop_count,select_root,mandatory_plan
from pathveil.rho_adaptive.workflow import validate_assets
from pathveil.rho_adaptive.construction import Adapter


class RhoAdaptiveCounts(unittest.TestCase):
    def test_formula_anchors_and_intervals(self):
        for rho,wanted in [(1,3),(1.99,3),(2,4),(2.5,4),(7.99,4),(8,5),
                           (12.775510204081632,5),(16.95,5),(17.0123,5),(32,6)]:
            self.assertEqual(high_secondhop_count(rho),wanted)

    def test_no_default_for_unknown_or_invalid_structure(self):
        for rho in [float("nan"),float("inf"),-1,0,.5]:
            with self.assertRaises(ValueError):high_secondhop_count(rho)
        for rho in [None,float("nan"),-1]:
            with self.assertRaises(ValueError):select_root({},rho)

    def test_root_union_keeps_both_models_not_first_three_combined(self):
        original=dict(method="m",top3_union_sources={"v1":[2,3,4],"struct":[4,5,6]},
            selected_neighbors=[2,3,4,5,6,7],root=dict(rankings={"m":[4,5,2,3,6,7]},
            selected_neighbors=[2,3,4,5,6,7],support_probabilities={str(n):.8 for n in range(2,8)},
            provenance={str(n):dict(selected=True) for n in range(2,8)}))
        before=copy.deepcopy(original);new=select_root(original,17.)
        self.assertEqual(new["selected_neighbors"],[4,5,2,3,6])
        self.assertFalse(new["root"]["provenance"]["7"]["selected"])
        self.assertEqual(original,before)
        self.assertEqual(select_root(original,1.),original)
        self.assertEqual(LOW_ROOT_LIMIT,21)


class ForcedSourceRecursion(unittest.TestCase):
    def test_all_roots_expand_even_with_adequate_tails(self):
        calls=[]
        def predict(prefix):
            calls.append(prefix)
            return dict(candidates=[dict(asn=n,admitted=True) for n in ["1",prefix[1],"7","8","9","10"]])
        result=mandatory_plan("1",["2","3"],lambda p,k:dict(valid_tail_count=300,count_is_lower_bound=True),predict,3)
        self.assertEqual(calls,[("1","2"),("1","3")])
        self.assertEqual(result["prefixes"],[("1","2","7"),("1","2","8"),("1","2","9"),
                                              ("1","3","7"),("1","3","8"),("1","3","9")])
        self.assertTrue(all(len(p)==3 for p in result["prefixes"]))

    def test_sparse_valid_choices_are_not_padded_with_unsupported_nodes(self):
        result=mandatory_plan("1",["2"],lambda p,k:dict(valid_tail_count=0),
            lambda p:dict(candidates=[dict(asn="3",admitted=False),dict(asn="4",admitted=True),dict(asn="4",admitted=True)]),5)
        self.assertEqual(result["prefixes"],[("1","2","4")])
        self.assertTrue(result["no_broad_root_tail_bypass"])

    def test_no_valid_child_leaves_source_unresolved_without_broad_tail_fallback(self):
        result=mandatory_plan("1",["2"],lambda p,k:dict(valid_tail_count=300),lambda p:dict(candidates=[]),5)
        self.assertEqual(result["prefixes"],[])
        self.assertEqual(result["decisions"][0]["stop_reason"],"no_supported_unvisited_next_hop")


class FrozenProtocolBoundaries(unittest.TestCase):
    def fixture(self):
        date="20260301"
        registry=dict(mode="bgp",date=date,routing_date=date,auxiliary_dates=dict(asrank=date),test_sources=["9"])
        proto=dict(date=date,sources=["9"]);data=dict(date=date,routing_date=date)
        upstream=dict(date=date);bundle=dict(date=date,model_fit_date=date,training_sources=["1"],formal_labels_read=False)
        model=copy.deepcopy(bundle)
        return date,registry,proto,data,upstream,bundle,model

    def test_registered_same_date_source_isolated_models_are_accepted(self):
        validate_assets(*self.fixture())

    def test_model_from_nearby_date_is_rejected(self):
        fields=list(self.fixture());fields[-1]["model_fit_date"]="20260201"
        with self.assertRaisesRegex(ValueError,"Cross-date"):
            validate_assets(*fields)

    def test_final_source_cannot_be_in_fitting_sources(self):
        fields=list(self.fixture());fields[-1]["training_sources"]=["9"]
        with self.assertRaisesRegex(ValueError,"Formal source"):
            validate_assets(*fields)

    def test_low_rho_adapter_delegates_to_original_tail_capacity_protocol(self):
        adapter=Adapter.__new__(Adapter);adapter.rho=.5
        with patch("pathveil.rho_adaptive.construction.SafeFrontierAdapter.plan",return_value={"original":True}) as original:
            self.assertEqual(adapter.plan(),{"original":True})
            original.assert_called_once_with()


if __name__=="__main__":unittest.main()
