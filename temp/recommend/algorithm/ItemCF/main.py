if __name__ == '__main__':
    from temp.recommend.data import data_loader, data_process
    from temp.recommend.algorithm.ItemCF.tool import item_similarity, user_item_score
    from temp.recommend.algorithm.common import topk

    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k, negative_sample_ratio=0)

    W = item_similarity(train_data, n_user, n_item)
    scores = user_item_score(train_data, n_user, n_item, W, N=10)

    score_fn = lambda ui: [scores[u][i] for u, i in zip(ui['user_id'], ui['item_id'])]
    topk(topk_data, score_fn)
