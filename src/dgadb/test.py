from src.dgadb.storage import TemporalGraphLoader
from src.dgadb.preprocessing.pipeline import DataLoader, Pipeline, TemporalSplitter, TimestampNormalizer, StructureNormalizer
import torch
from src.dgadb.preprocessing import AnomalyInjector
import pandas as pd
import os

if __name__ == "__main__":
    dataset_names = [
        "bitcoin-alpha",
        "bitcoin-otc",
        "uc-social",
        "as-topology",
        "epinions",
        "email-dnc",
        "digg-homo"
    ]

    split_ratio = .7

    splitter = TemporalSplitter(train_ratio=split_ratio)
    timestamp_norm = TimestampNormalizer()

    anom_ratios = [.01, .05, .1]

    for name in dataset_names:
        data_loader = DataLoader("data/"+name)
        pipeline = Pipeline([data_loader, splitter, timestamp_norm])

        cont = pipeline.run()
        tg = cont.to_temporal_graph()

        # if tg.edge_labels is None:
        #     tg.edge_labels = torch.

        print(tg.edges.shape)

        anom_inj = AnomalyInjector(tg)
        for r in anom_ratios:
            anom_tg = anom_inj.generate_anomalous_samples(
                "s", anom_test_ratio=r, anom_train_ratio=0.5)

            anom_train_mask = anom_tg.train_mask
            anom_test_mask = anom_tg.test_mask

            # train
            anom_train_edges = anom_tg.edges[anom_train_mask]
            anom_train_t = anom_tg.t[anom_train_mask].unsqueeze(1)
            anom_train_g = torch.cat([anom_train_edges, anom_train_t], dim=1)

            # test
            anom_test_edges = anom_tg.edges[anom_test_mask]
            anom_test_t = anom_tg.t[anom_test_mask].unsqueeze(1)

            if anom_tg.edge_labels is None:
                print("---------------------------------------------")
                print(
                    f"Couldn't generate anomalies for dataset: {name}, anomaly rate: {r}")
                print("---------------------------------------------")
                continue

            # assert anom_tg.edge_labels is not None, f"dataset: {name}, anomaly rate: {r}"
            anom_test_labels = anom_tg.edge_labels[anom_test_mask].unsqueeze(1)

            anom_test_g = torch.cat(
                [anom_test_edges, anom_test_t, anom_test_labels], dim=1)

            save_dir = f"anom_gen/{name}_{split_ratio}_{split_ratio}_{r}"
            os.makedirs(save_dir, exist_ok=True)

            df_train = pd.DataFrame(
                anom_train_g.numpy(), columns=["src", "tgt", "t"])
            df_train.to_csv(f"{save_dir}/train.csv", index=False)

            df_test = pd.DataFrame(anom_test_g.numpy(), columns=[
                                   "src", "tgt", "t", "label"])
            df_test.to_csv(f"{save_dir}/test.csv", index=False)

        #     break

        # break
