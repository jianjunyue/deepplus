if __name__ == '__main__':
    import temp.recommend.utility.gpu_memory_growth
    from temp.recommend.data import data_loader, data_process
    from temp.recommend.algorithm.FM.model import FM_model
    from temp.recommend.algorithm.train import train

    n_user, n_item, train_data, test_data, topk_data = data_process.pack(data_loader.ml100k)

    model = FM_model(n_user, n_item, dim=16, l2=1e-6)

    train(model, train_data, test_data, topk_data, epochs=10, batch=512)
