"""Generate train-extra GMPI rows using the released pool for both consumers."""
import argparse
from pathlib import Path

from .common import (
    audit_pool_consumers,
    assert_source_held_out,
    digest,
    install_label_guard,
    json_write,
    load_contract,
    project_imports,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--support-data-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--source-asn", default="63835")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--prefix-tree-dir", type=Path)
    parser.add_argument("--pathgen-workers", type=int, default=1)
    args = parser.parse_args()
    contract = load_contract(args.data_manifest, args.release, args.source_asn)
    args.outdir.mkdir(parents=True, exist_ok=False)
    project_imports(args.data_manifest)
    install_label_guard(args.outdir)
    import environment
    import getData

    env = environment.Env(contract["date"], contract["date"])
    holdout = assert_source_held_out(env, args.source_asn)
    env.support_data_dir = str(args.support_data_dir.resolve()) + "/"
    env.graph_selected_neighbors_csv = contract["selected_neighbors_csv"]
    env.graph_recursive_anchors_csv = contract["recursive_anchors_csv"]
    env.augment_graph_from_pipeline_priors()
    env.k = env.observed_path_num = args.k
    if args.pathgen_workers < 1:
        parser.error("pathgen-workers must be positive")
    env.top_n, env.num_thread_override = 1, args.pathgen_workers
    env.as_type, env.pathGenAlgorithm = "devoid", "PG"
    env.train_extra_data_mode = env.pg_local_knowledge_mode = "augmentation_only"
    env.isGetData = env.refresh_kpath_cache = True
    env.result_dir_override = str(args.outdir / "pathgen_results")
    env.meta_result_dir_override = str(args.outdir / "pathgen_meta_results")
    env.analyze_data_dir_override = str(args.outdir / "model_input_data_PG") + "/"
    env.curDate_baseDir = str(args.outdir / "runtime_cache") + "/"
    Path(env.curDate_baseDir).mkdir()
    before = audit_pool_consumers(env, args.source_asn, contract)
    if args.prefix_tree_dir is not None:
        from pathveil.prefix_tree.traceroute import install_generator
        install_generator(args.prefix_tree_dir, args.data_manifest, contract, args.outdir)
    getData.main_extra_training(env, [args.source_asn])
    after = audit_pool_consumers(env, args.source_asn, contract)
    output = (
        args.outdir
        / "model_input_data_PG/train/meta-devoid"
        / (args.source_asn + "_train_extra_data_generated_data.csv")
    )
    if not output.is_file() or not output.stat().st_size:
        raise ValueError("No training rows were generated from the released pool")
    json_write(
        args.outdir / "TRAINING_INPUTS.json",
        {
            "generated_csv": str(output.resolve()),
            "generated_sha256": digest(output),
            "pool_contract": contract,
            "before": before,
            "after": after,
            "source_holdout_audit": holdout,
            "traceroute_labels_read": False,
        },
    )


if __name__ == "__main__":
    main()
