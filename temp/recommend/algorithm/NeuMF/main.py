if __name__ == '__main__':
    import temp.recommend.utility.gpu_memory_growth
    from temp.recommend.algorithm.NeuMF.train import train_with_pretrain, train_without_pretrain
    from temp.recommend.data import data_loader, data_process

    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k)

    gmf_dim, mlp_dim, layers, l2 = 8, 32, [32, 16, 8], 0

    train_with_pretrain(n_user, n_item, train_data, test_data, topk_data, gmf_dim, mlp_dim, layers, l2)
    print('---------------------------------------------------------------------------------------------')
    train_without_pretrain(n_user, n_item, train_data, test_data, topk_data, gmf_dim, mlp_dim, layers, l2)
