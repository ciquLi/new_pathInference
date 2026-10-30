import unittest

from pathveil.rho_adaptive.full_tree import POLICY,validate_same_generator,numerical_tree


class ConsistentGeneratorProtocol(unittest.TestCase):
    def test_both_rho_groups_have_the_same_registered_generator(self):
        self.assertEqual(POLICY["training_generators"],{"low":"Tree B + PG","high":"Tree B + PG"})

    def test_a_shortened_query_tree_cannot_pass_as_full_tree(self):
        ready=dict(training_generator="Tree B + PG")
        with self.assertRaisesRegex(ValueError,"lost selected"):
            validate_same_generator(ready,dict(primary_leaves=[dict(prefix=["1","2"])],fallback_leaves=[]),[["1","2","3"]])

    def test_legacy_training_pg_cannot_pass_the_new_protocol(self):
        with self.assertRaisesRegex(ValueError,"generators differ"):
            validate_same_generator(dict(training_generator="PG"),{},[])

    def test_phase_duplicates_preserve_the_full_physical_prefix_set(self):
        tree=dict(primary_leaves=[dict(prefix=["1","2","3"],phase_state="UP"),
                                  dict(prefix=["1","2","3"],phase_state="DOWN")],fallback_leaves=[])
        validate_same_generator(dict(training_generator="Tree B + PG"),tree,[["1","2","3"]])

    def test_cache_equivalence_includes_tiers_phase_states_and_weights(self):
        one=dict(primary_leaves=[dict(prefix=["1","2"],structural_weight=1.,phase_state="UP")],fallback_leaves=[],prefix_classification=[])
        two=dict(one,primary_leaves=[dict(prefix=["1","2"],structural_weight=1.,phase_state="DOWN")])
        self.assertNotEqual(numerical_tree(one),numerical_tree(two))


if __name__=="__main__":unittest.main()
