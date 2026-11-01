import torch
import torch.nn.functional as F
import torch.optim as optim
import GMPI_pathLikelihoodEst as attNet
import os
import time
import myDataset
import pandas as pd
import numpy as np
import tqdm

# from tensorboardX import SummaryWriter


# 用于因为batch_size增加维度的情况
def score_paths(pathScore, similarity, n):
    l = pathScore.size(1)
    N = n if n <= l else l
    _, idx = torch.topk(pathScore, N, -2)
    rwdTensor = similarity.gather(1, idx)
    max_rwd, max_index = torch.max(rwdTensor, dim=-2, keepdim=False)
    return max_rwd


def train(model, dataset, optimizer, env, epoch, str_code, num_infer=1):
    print_num, loader_params = env.training_print_num, env.train_loader_params
    model.train()
    total_loss = 0.0
    total_score = 0.0
    total_uba = 0.0
    count = 0
    exact_count = 0
    loader = myDataset.DataLoaderX(dataset, **loader_params)
    data_iter = tqdm.tqdm(
        enumerate(loader),
        desc="EP_%s:%d" % (str_code, epoch),
        total=len(loader),
        bar_format="{l_bar}{r_bar}",
    )
    segScore = 0.0
    segLoss = 0.0
    uba = 0.0
    seg_exact_count = 0
    for idx, item in data_iter:
        # for idx,item in enumerate(loader):
        for i in range(len(item) - 3):
            item[i] = torch.as_tensor(item[i]).cuda()
        (
            kpaths,
            k_last_second_hops,
            kpathsAP,
            hislengthAP,
            hisPaths,
            his_last_second_hops,
            hisPathsAP,
            similarity,
            ap,
            _,
            _,
        ) = item

        # Name the history inputs: full paths and penultimate hops have
        # different roles even though broadcasting can hide a swap.
        pathScore = model(
            kpathsEmbedding=kpaths,
            k_lastSecHops=k_last_second_hops,
            kcurASPair=kpathsAP,
            hcurASPair=hislengthAP,
            hispathEmbedding=hisPaths,
            his_lastSecHops=his_last_second_hops,
            hisASPairsEmbedding=hisPathsAP,
        )
        score = score_paths(pathScore, similarity, num_infer)

        loss = F.smooth_l1_loss(pathScore, similarity.detach())
        total_loss += loss.item()
        segLoss = loss.item()
        segScore = torch.sum(score).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # count += kcurASPair.size(0)
        maxSim, _ = torch.max(similarity, dim=1)
        maxSim = torch.sum(maxSim).item()
        uba = maxSim
        exact = score[torch.where(score >= 1.0)]
        seg_exact_count = exact.size(0)

        curCount = kpaths.size(0)
        count += curCount
        total_score += segScore
        exact_count += seg_exact_count
        total_uba += uba

        if idx % print_num == 0 and idx > 0:
            # curCount = print_num*loader_params['batch_size']
            # count += curCount
            # total_score += segScore
            # exact_count += seg_exact_count
            # total_uba += uba
            print(
                "count:",
                curCount,
                "segLoss:",
                segLoss,
                "segScore:",
                segScore,
                "uba:",
                uba,
                "segAcc:",
                segScore / curCount,
                "seg_exact_count:",
                seg_exact_count,
                "seg_exact_ratio:",
                seg_exact_count / curCount,
            )
            # print(curCount,segLoss,segScore,uba,segScore/curCount,seg_exact_count,seg_exact_count/curCount)
            # segScore = 0.0
            # segLoss = 0.0
            # uba = 0.0
            # seg_exact_count = 0
    loader = None
    return total_loss, total_score, count, total_uba


def test(model, dataset, optimizer, env, epoch, str_code, num_infer=1):
    print_num, loader_params = env.training_print_num, env.test_loader_params
    model.eval()
    total_loss = 0.0
    total_score = 0.0
    total_uba = 0.0
    count = 0
    apcount = 0
    exact_count = 0
    loader = myDataset.DataLoaderX(dataset, **loader_params)
    data_iter = tqdm.tqdm(
        enumerate(loader),
        desc="EP_%s:%d" % (str_code, epoch),
        total=len(loader),
        bar_format="{l_bar}{r_bar}",
    )
    segScore = 0.0
    segLoss = 0.0
    uba = 0.0
    seg_exact_count = 0
    ap_score = []
    with torch.no_grad():
        for idx, item in data_iter:
            # Keep epoch-level validation bounded by complete batches.  Final
            # test_multiple evaluation remains uncapped.
            max_batches = 0
            if str_code == "test":
                max_batches = max(0, int(getattr(env, "epoch_test_batches", 19)))
                if max_batches and idx >= max_batches:
                    break
            for i in range(len(item) - 3):
                item[i] = torch.as_tensor(item[i]).cuda()

            (
                kpaths,
                k_last_second_hops,
                kpathsAP,
                hislengthAP,
                hisPaths,
                his_last_second_hops,
                hisPathsAP,
                similarity,
                ap,
                generatedPathsNum,
                hisPathsNum,
            ) = item
            pathScore = model(
                kpathsEmbedding=kpaths,
                k_lastSecHops=k_last_second_hops,
                kcurASPair=kpathsAP,
                hcurASPair=hislengthAP,
                hispathEmbedding=hisPaths,
                his_lastSecHops=his_last_second_hops,
                hisASPairsEmbedding=hisPathsAP,
            )

            score = score_paths(pathScore, similarity, num_infer)

            loss = F.smooth_l1_loss(pathScore, similarity.detach())
            total_loss += loss.item()
            segLoss += loss.item()
            segScore = torch.sum(score).item()
            maxSim, _ = torch.max(similarity, dim=1)
            maxSim = torch.sum(maxSim).item()
            uba = maxSim
            exact = score[torch.where(score >= 1.0)]
            seg_exact_count = exact.size(0)

            # convert score to list
            score_list = score.tolist()
            generatedPathsNum = generatedPathsNum.tolist()
            hisPathsNum = hisPathsNum.tolist()
            # save the score of each ap
            ap_score.extend(
                [
                    [ap[i], score_list[i][0], generatedPathsNum[i], hisPathsNum[i], num_infer]
                    for i in range(len(ap))
                ]
            )
            curCount = kpaths.size(0)
            count += curCount
            total_score += segScore
            exact_count += seg_exact_count
            total_uba += uba

            if idx % print_num == 0 and idx > 0:
                print(
                    curCount,
                    segLoss,
                    segScore,
                    uba,
                    segScore / curCount,
                    seg_exact_count,
                    seg_exact_count / curCount,
                )

    loader = None
    if str_code == "test_multiple":
        saving_dir = "result/%s/%s" % (env.parent, "PerAPScores")
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        ap = ap_score[0][0]
        vas = env.cur_vas
        saving_path = saving_dir + "/%s-%.2f-%s.csv" % (vas, env.meta_split_num, env.as_type)
        # save the score of each ap to a csv file
        df = pd.DataFrame(
            ap_score, columns=["ap", "score", "generatedPathsNum", "hisPathsNum", "num_infer"]
        )
        df.to_csv(saving_path, index=False, mode="a")
    return total_loss, total_score, count, apcount, exact_count, total_uba


def build_lr_scheduler(optimizer, env):
    """Return the configured validation-driven scheduler, or None."""
    scheduler_name = getattr(env, "lr_scheduler", "none")
    if scheduler_name == "none":
        return None
    if scheduler_name != "plateau":
        raise ValueError("Unsupported learning-rate scheduler: %s" % scheduler_name)
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=env.lr_scheduler_factor,
        patience=env.lr_scheduler_patience,
        threshold=env.lr_scheduler_threshold,
        threshold_mode="abs",
        min_lr=env.lr_scheduler_min_lr,
    )


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def main(env, version, desc, train_epi, train_data_file, test_data_file):
    torch.cuda.set_device(0)
    k = env.k
    hiddedNum = env.hiddedNum
    graphEmbeddingLen = env.node2vecOutputDim
    asrankFeatureDim = env.asrankFeatureDim
    lr = env.lr
    nodeEmbeddingLen = graphEmbeddingLen + asrankFeatureDim
    pathEmbeddingLength = nodeEmbeddingLen * 3  # pathEmbeddingLength
    pairEmbeddingLen = nodeEmbeddingLen * 4

    # maxLen
    # existPath = env.getExistPath('train')
    # maxLen = env.getMaxLen(existPath)
    # print('maxLen:%d'%maxLen)
    # existPath = {}
    # return
    # maxLenDict = {'20211002':15,'20210901':15,'20211001':15,'20210801':17,'20211003':15,'20231001':20}
    # maxLen = maxLenDict[env.parent]

    maxLen = 20

    config = {}
    config["hiddedNum"] = env.hiddedNum
    config["graphEmbeddingLen"] = env.node2vecOutputDim
    config["asrankFeatureDim"] = env.asrankFeatureDim
    config["lr"] = env.lr
    config["local_lr"] = env.local_lr
    config["nodeEmbeddingLen"] = config["graphEmbeddingLen"] + config["asrankFeatureDim"]
    config["pathEmbeddingLength"] = config["nodeEmbeddingLen"] * 3  # pathEmbeddingLength
    config["pairEmbeddingLen"] = config["nodeEmbeddingLen"] * 4
    config["nodeEmbeddingGrad"] = env.nodeEmbeddingGrad
    config["maxLen"] = maxLen

    result_dir = getattr(env, "training_result_dir_override", "") or (
        "result/%s/%s" % (env.parent, env.pathGenAlgorithm)
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    resultFile = "%s/%s_result.txt" % (result_dir, version)
    lossFile = "%s/%s_loss.txt" % (result_dir, version)

    fp = open(resultFile, "a")
    fp.write("desc:%s\n" % (desc))
    fp.write(version + "\n")
    fp.write("maxLen:%d\n" % (maxLen))

    fp.close()
    fp = open(lossFile, "a")
    fp.write(version + "\n")
    fp.close()

    model_dir = env.baseDir + "model/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    att_path = os.path.join(model_dir, "%s_%d_%d_attNetwork.pkl" % (version, hiddedNum, k))
    load_existing_model = bool(getattr(env, "load_existing_model", False))
    if load_existing_model and os.path.exists(att_path):
        attNetwork = torch.load(att_path)
        if getattr(attNetwork, "input_semantics_version", 1) != 2:
            raise ValueError(
                "Existing checkpoint uses legacy input semantics; train a new "
                "model instead of resuming it with corrected inputs"
            )
        print("load model successfully")
    else:
        node_ASVectors = env.get_node_vectors()
        ASVectors = torch.from_numpy(np.array(node_ASVectors)).float()
        attNetwork = attNet.attentionBasedScoreNetwork(
            ASVectors, pathEmbeddingLength, nodeEmbeddingLen, pairEmbeddingLen, hiddedNum, maxLen
        ).cuda()

        attNetwork.asEmbedding.weight.requires_grad = True
        if os.path.exists(att_path):
            print("ignore existing model and create a new model")
        else:
            print("create model successfully")
        # Persist the fresh initialization immediately so this run cannot
        # accidentally reload an older checkpoint during final validation.
        torch.save(attNetwork, att_path)

    optimizer = optim.Adam(attNetwork.parameters(), lr=lr)
    scheduler = build_lr_scheduler(optimizer, env)

    parameters_pr = sum([p.nelement() for p in attNetwork.pr.parameters()])
    parameters_pm = sum([p.nelement() for p in attNetwork.pm.parameters()])
    ratio = parameters_pr / parameters_pm
    print("parameters comparison:", round(parameters_pr / parameters_pm, 2))
    print("Total Parameters:", sum([p.nelement() for p in attNetwork.parameters()]))

    # writer_logdir = 'runs/%s'%(version)
    # writer = SummaryWriter(log_dir=writer_logdir)
    # writer2 = SummaryWriter(comment=version+'_2')
    # writer3 = SummaryWriter(comment=version+'_3')

    buffersize = 60000
    seed = 1
    train_dataset = myDataset.IterableDatasetShuffle(
        train_data_file, env, maxLen, k, buffersize, seed, False
    )
    test_dataset = myDataset.IterableDatasetShuffle(
        test_data_file, env, maxLen, k, buffersize, seed, False
    )
    # train_extra_dataset = myDataset.IterableDatasetShuffle(train_extra_file, env, maxLen, k, buffersize, seed, False)
    count = 0  # the epochs that accuracy is not improved
    bestTest = 0
    n_epi = 0
    for n_epi in range(train_epi):
        t1 = time.time()
        print("train----------------------------")
        num_infer = 1
        lr = current_lr(optimizer)
        total_loss, total_scores, train_count, total_uba = train(
            attNetwork, train_dataset, optimizer, env, n_epi, "train", num_infer
        )

        train_acc = total_scores / train_count
        uba = total_uba / train_count

        # writer.add_scalar('train_loss', total_loss, n_epi)
        # writer.add_scalar('train_acc', train_acc, n_epi)
        # writer.add_scalar('train_count_score', train_count, n_epi)
        # writer.add_scalar('train_total_uba', total_uba, n_epi)
        # writer.add_scalar('train_score', total_scores, n_epi)
        # writer2.add_scalar('train_count_score', total_scores, n_epi)    #to make the two scalar in the same figure

        fp = open(resultFile, "a")
        fp.write(
            "hiddenDim=%d,k=%d,observed_path_num=%d,maxLen=%d,lr=%.8f,batchSize=%d\n"
            % (
                hiddedNum,
                k,
                env.observed_path_num,
                maxLen,
                lr,
                env.train_loader_params["batch_size"],
            )
        )
        txt = "epi:%d,ObservedNum:%d,trainDataNum:%d,total_rwd:%.3f,ratio:%.3f,uba:%.3f" % (
            n_epi,
            env.observed_path_num,
            train_count,
            total_scores,
            train_acc,
            uba,
        )
        fp.write(txt + "\n")
        fp.close()

        fp = open(lossFile, "a")
        fp.write("%.5f\n" % (total_loss))
        fp.close()

        print("test----------------------------")
        num_infer = 1
        test_loss, total_testScore, total_testCount, apcount, exact_count, total_uba = test(
            attNetwork, test_dataset, optimizer, env, n_epi, "test", num_infer
        )

        test_acc = total_testScore / total_testCount
        uba = total_uba / total_testCount
        # test_esr = exact_count/total_testCount
        # writer.add_scalar('test_acc', test_acc, n_epi)
        # writer.add_scalar('test_esr', test_esr, n_epi)
        # writer.add_scalar('test_count_score_exact', total_testCount, n_epi)
        # writer2.add_scalar('test_count_score_exact', total_testScore, n_epi)
        # writer3.add_scalar('test_count_score_exact', exact_count, n_epi)

        txt = "epi:%d,testDataNum:%d,total_rwd:%.3f,accuracy:%.3f,uba:%.3f,N=%d" % (
            n_epi,
            total_testCount,
            total_testScore,
            test_acc,
            uba,
            num_infer,
        )
        fp = open(resultFile, "a")
        fp.write(txt + "\n")

        t2 = time.time()
        fp.write("%.3f\n" % (t2 - t1))
        print(t2 - t1)

        if test_acc > bestTest:
            bestTest = test_acc
            torch.save(attNetwork, att_path)  # the local model
            fp.write("best_epi:%d\n" % (n_epi))
            count = 0
        else:
            count += 1
        previous_lr = current_lr(optimizer)
        if scheduler is not None:
            scheduler.step(test_acc)
            lr = current_lr(optimizer)
            if lr < previous_lr:
                fp.write("lr_reduced:%.8f->%.8f\n" % (previous_lr, lr))
                print("LR is reduced from {} to {}".format(previous_lr, lr))
                count = 0
        fp.close()
        if scheduler is None and count >= env.early_stop_patience:
            break
        if (
            scheduler is not None
            and current_lr(optimizer) <= env.lr_scheduler_min_lr
            and count >= env.early_stop_patience
        ):
            break

    print("test multiple----------------------------")
    if os.path.exists(att_path):
        attNetwork = torch.load(att_path)
        print("load best model successfully")

    inferNum = 1  # 推荐的条数
    for i in range(inferNum):
        num = i + 1
        (
            loss,
            total_validScore,
            total_validCount,
            total_probinfer_aps,
            total_exact_count,
            total_uba,
        ) = test(attNetwork, test_dataset, optimizer, env, n_epi, "test_multiple", num)

        ratio = total_validScore / total_validCount
        uba = total_uba / total_validCount
        txt = (
            "epi:%d,validateDataNum:%d,total_rwd:%.3f,accuracy:%.3f,uba:%.3f,apcount:%d,exact_count:%d,N=%d"
            % (
                n_epi,
                total_validCount,
                total_validScore,
                ratio,
                uba,
                total_probinfer_aps,
                total_exact_count,
                num,
            )
        )
        fp = open(resultFile, "a")
        dataTxt = "test+validate\n"
        fp.write(dataTxt)
        fp.write(txt + "\n")
        fp.close()
