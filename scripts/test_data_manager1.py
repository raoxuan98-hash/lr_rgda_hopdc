import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_manager1 import IncrementalDataManager


def main():
    # Choose a dataset that exists under utils.data1 ROOT
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    dm = IncrementalDataManager(
        dataset_name=dataset_name,
        initial_classes=5,
        increment_classes=1,
        shuffle=True,
        seed=42)

    print({
        "dataset": dataset_name,
        "num_classes": dm.num_classes,
        "nb_tasks": dm.nb_tasks,
        "class_order_head": dm.class_order[:10],
    })

    # Print task class ranges
    for t in range(dm.nb_tasks):
        new_cls = dm.get_task_classes(t, cumulative=False)
        cum_cls = dm.get_task_classes(t, cumulative=True)
        print({
            "task": t,
            "new": len(new_cls),
            "cum": len(cum_cls),
            "new_head": new_cls[:5],
        })

    # Subset length sanity
    train0 = dm.get_subset(0, "train", cumulative=False)
    train1cum = dm.get_subset(1 if dm.nb_tasks > 1 else 0, "train", cumulative=True)
    print({
        "subset_lens": {
            "train_t0": len(train0),
            "train_t1_cum": len(train1cum),
        }
    })


if __name__ == "__main__":
    main()
