from __future__ import annotations


def _k_tag(value: int) -> str:
    return f"{value // 1000}k"


def dataset_tag(dataset_name: str, class_names: list[str]) -> str:
    class_tag = "-".join(class_names)
    return f"{dataset_name}_{class_tag}"


def embedding_filename(
    dataset_tag_name: str,
    window_size: int,
    step_size: int,
    layer: int,
    split: str,
) -> str:
    return (
        f"{dataset_tag_name}_window{_k_tag(window_size)}_step{_k_tag(step_size)}"
        f"_layer{layer}_{split}.pt"
    )


def rf_model_filename(
    dataset_tag_name: str,
    train_window_size: int,
    train_step_size: int,
    layer: int,
) -> str:
    return (
        f"{dataset_tag_name}_window{_k_tag(train_window_size)}"
        f"_step{_k_tag(train_step_size)}_layer{layer}_train.rf.pkl"
    )


def rf_test_result_filename(
    dataset_tag_name: str,
    test_window_size: int,
    test_step_size: int,
    layer: int,
    threshold: float,
) -> str:
    return (
        f"{dataset_tag_name}_window{_k_tag(test_window_size)}"
        f"_step{_k_tag(test_step_size)}_layer{layer}_test_thr{threshold:.2f}.rf.tsv"
    )


def result_test_dataset_name(
    dataset_tag_name: str,
    test_window_size: int,
    test_step_size: int,
) -> str:
    return (
        f"{dataset_tag_name}_window{_k_tag(test_window_size)}"
        f"_step{_k_tag(test_step_size)}_test"
    )