import torch

def scores_to_ranks(scores, descending=False):
    return torch.argsort(scores, dim=1, descending=descending) + 1

def get_gt_ranks(ranks, ans_idx):
    batch_size = ans_idx.size(0)
    gt_ranks = torch.Tensor(batch_size)
    for i in range(batch_size):
        gt_ranks[i] = ranks[i, ans_idx[i]]

    return gt_ranks

def get_recall(gt_ranks, top_k=1):
    num_ques = gt_ranks.size(0)
    return float(torch.sum(gt_ranks <= top_k).item()) / num_ques

def get_mean(gt_ranks):
    return torch.mean(gt_ranks)

def process_results(gt_ranks, question_types):

    gt_ranks_s = gt_ranks[question_types == 0]
    gt_ranks_av = gt_ranks[question_types == 1]

    print("Number of questions per types...")
    print("Spatial : {}, Audio-visual : {}, Full : {}".format(
        torch.sum(question_types == 0).item(),
        torch.sum(question_types == 1).item(),
        question_types.size(0))
    )

    for i in [1, 5, 10]:
        print(f"Top {i} recall...")
        print("Spatial : {}, Audio-visual : {}, Full : {}".format(
                get_recall(gt_ranks_s, i),
                get_recall(gt_ranks_av, i),
                get_recall(gt_ranks, i)
            )
        )

    print("Mean rank...")
    print("Spatial : {}, Audio-visual : {}, Full : {}".format(
            get_mean(gt_ranks_s),
            get_mean(gt_ranks_av),
            get_mean(gt_ranks)
        )
    )
    