import environment
import argparse
import os
import commonFunction as commonFun
import getData as gd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_num", default=1, type=int)
    parser.add_argument("--cur_idx", default=0, type=int)
    parser.add_argument("--date", required=True, type=str)
    parser.add_argument("--k", default=200, type=int)
    parser.add_argument("--oppNum", default=200, type=int)
    parser.add_argument("--meta_test_as_num", default=10, type=int)
    parser.add_argument("--as_type", default="devoid", type=str)
    parser.add_argument("--target_vases", default="", type=str)
    parser.add_argument("--num_thread", default=0, type=int)
    parser.add_argument(
        "--extra_training_only",
        action="store_true",
        help="Regenerate only P95/P99-dependent extra-training inputs.",
    )
    parser.add_argument(
        "--reuse_test_data",
        action="store_true",
        help="Reuse existing test generated-data files while rebuilding train inputs.",
    )
    parser.add_argument("--prefix_level", action="store_true")
    parser.add_argument("--max_dests", default=0, type=int)
    parser.add_argument("--dgi_threshold", default=4.0, type=float)
    parser.add_argument(
        "--pg_local_knowledge_mode",
        default="augmentation_only",
        choices=(
            "augmentation_only",
            "augmentation_plus_external_observation",
            "merged_structure",
            "merged",
        ),
    )
    parser.add_argument(
        "--wg-support-data-mode",
        default="neighbor_competition",
        choices=(
            "augmentation_only",
            "augmentation_plus_external_observation",
            "neighbor_competition",
        ),
        help="WG root competition input; default merges sources per neighbor and destination.",
    )
    parser.add_argument(
        "--train-extra-data-mode",
        default="neighbor_competition",
        choices=(
            "augmentation_only",
            "augmentation_plus_external_observation",
            "neighbor_competition",
        ),
        help="Truth paths used to build unchanged GlobalPath train-extra inputs.",
    )
    parser.add_argument("--graph_selected_neighbors_csv", default="")
    parser.add_argument("--graph_recursive_anchors_csv", default="")
    parser.add_argument(
        "--support_data_dir",
        default="",
        help="Optional augmented-path support_data directory for an isolated run.",
    )
    parser.add_argument(
        "--experiment_dir",
        default="",
        help="Optional isolated directory for generated data and pathgen results.",
    )
    parser.add_argument(
        "--path_gen_algorithm",
        default="PG",
        choices=("PG",),
        help="PG uses the PA-GMPI personalized generator; HFVG is its legacy name.",
    )
    args = parser.parse_args()
    split_num = args.split_num
    cur_idx = args.cur_idx
    date = args.date
    k = args.k
    m = args.oppNum
    meta_test_as_num = args.meta_test_as_num
    as_type = args.as_type
    target_vases = [item.strip() for item in args.target_vases.split(",") if item.strip()]
    num_thread = args.num_thread
    prefix_level = args.prefix_level
    max_dests = args.max_dests
    dgi_threshold = args.dgi_threshold
    pg_local_knowledge_mode = args.pg_local_knowledge_mode
    wg_support_data_mode = args.wg_support_data_mode
    train_extra_data_mode = args.train_extra_data_mode
    path_gen_algorithm = args.path_gen_algorithm
    env = environment.Env(date, date)
    if args.support_data_dir:
        env.support_data_dir = os.path.join(os.path.abspath(args.support_data_dir), "")
    if args.experiment_dir:
        experiment_dir = os.path.abspath(args.experiment_dir)
        env.result_dir_override = os.path.join(experiment_dir, "pathgen_results")
        env.meta_result_dir_override = os.path.join(experiment_dir, "pathgen_meta_results")
        env.analyze_data_dir_override = os.path.join(
            experiment_dir, "model_input_data_%s" % path_gen_algorithm, ""
        )
    env.k = k
    env.top_n = 1
    env.observed_path_num = m
    env.as_type = as_type
    env.num_thread_override = num_thread
    env.reuse_test_data = args.reuse_test_data
    env.train_extra_data_mode = train_extra_data_mode
    vases = commonFun.select_meta_vases(date, meta_test_as_num, target_vases=target_vases)
    env.prefix_level = prefix_level
    env.max_dests = max_dests
    env.length_border = dgi_threshold
    env.pg_local_knowledge_mode = pg_local_knowledge_mode
    env.graph_selected_neighbors_csv = args.graph_selected_neighbors_csv
    env.graph_recursive_anchors_csv = args.graph_recursive_anchors_csv
    added_prior_edges = env.augment_graph_from_pipeline_priors()
    print("pipeline topology edges added:", added_prior_edges)
    if path_gen_algorithm == "WG":
        raise ValueError("The formal method supports PG only")
    splitted_vases = commonFun.split_list_evenly(vases, split_num)
    vases = splitted_vases[cur_idx]
    print("selected vases:", vases)
    if not args.extra_training_only:
        env.pathGenAlgorithm = path_gen_algorithm
        env.isGetData = True
        env.refresh_kpath_cache = True
        gd.main(env, vases)
        print("standard train/test data generation over")
    if as_type != "intensive":
        env.pathGenAlgorithm = path_gen_algorithm
        env.isGetData = True
        env.refresh_kpath_cache = True
        gd.main_extra_training(env, vases)
    print("extra-training data generation over")
