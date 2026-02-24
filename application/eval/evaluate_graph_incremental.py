import os
import re
import json
import hydra
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from hovsg.graph.graph import Graph
from hovsg.graph.object import Object
from hovsg.eval.hm3dsem_evaluator import HM3DSemanticEvaluator
from hovsg.utils.label_feats import get_label_feats

# pylint: disable=all


@hydra.main(version_base=None, config_path="../../config", config_name="eval_graph")
def main(params: DictConfig):

    # overwrite dataset path with the actual scene path
    dataset_path = os.path.join(params.main.dataset_path, params.main.split, params.main.scene_id)
    params.main.dataset_path = dataset_path

    # base directory where incremental graphs were saved
    save_dir = os.path.join(params.main.save_path, params.main.dataset, params.main.scene_id)

    # discover all frame_XXXXXX subdirectories, sorted by frame index
    frame_dirs = sorted(
        [d for d in os.listdir(save_dir) if re.match(r"frame_\d+", d) and os.path.isdir(os.path.join(save_dir, d))]
    )

    if len(frame_dirs) == 0:
        print(f"No incremental frame directories found under {save_dir}")
        return

    print(f"Found {len(frame_dirs)} incremental graph steps: {frame_dirs}")

    # load GT graph once
    evaluator_template = HM3DSemanticEvaluator(params)
    evaluator_template.load_gt_graph_from_json(os.path.join(dataset_path, "scene_info.json"))

    # load CLIP model and text feats once (Graph without pipeline config = eval mode, only loads CLIP)
    hovsg = Graph(params)
    text_feats, classes = get_label_feats(
        hovsg.clip_model, hovsg.clip_feat_dim,
        params.eval.obj_labels, params.main.save_path
    )

    # output directory for evaluation results
    eval_output_dir = os.path.join(save_dir, "eval_incremental")
    os.makedirs(eval_output_dir, exist_ok=True)

    # collect metrics across frames for plotting
    frame_indices = []
    hydra_precisions = []
    hydra_recalls = []
    hydra_recalls_all = []
    all_frame_metrics = {}

    for frame_dir_name in frame_dirs:
        # extract the frame index from the directory name
        frame_idx = int(re.search(r"frame_(\d+)", frame_dir_name).group(1))
        graph_path = os.path.join(save_dir, frame_dir_name, "graph")

        # check that the graph directory exists
        if not os.path.isdir(graph_path):
            print(f"Skipping {frame_dir_name}: no graph/ directory found")
            continue

        # check that rooms exist (needed for room evaluation)
        rooms_path = os.path.join(graph_path, "rooms")
        if not os.path.isdir(rooms_path) or len(os.listdir(rooms_path)) == 0:
            print(f"Skipping {frame_dir_name}: no rooms found in graph/")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {frame_dir_name} (frame index {frame_idx})")
        print(f"{'='*60}")

        # create a fresh evaluator with the same GT graph
        evaluator = HM3DSemanticEvaluator(params)
        evaluator.gt_graph = evaluator_template.gt_graph
        evaluator.gt_floors = evaluator_template.gt_floors
        evaluator.gt_rooms = evaluator_template.gt_rooms
        evaluator.gt_objects = evaluator_template.gt_objects

        # reset graph state and load the predicted graph for this frame
        hovsg.reset()
        hovsg.load_graph(path=graph_path)
        hovsg.graph.remove_node(0)

        # relabel objects based on the hm3dsem labels
        for node in hovsg.graph.nodes:
            if type(node) == Object:
                name = hovsg.identify_object(node.embedding, text_feats, classes)
                node.name = name

        # evaluate rooms (incremental=True: only visible GT rooms count for recall)
        evaluator.evaluate_rooms(hovsg.graph, incremental=True)

        # collect metrics
        room_metrics = evaluator.metrics.get("rooms", {})
        frame_indices.append(frame_idx)
        hydra_precisions.append(float(room_metrics.get("hydra_prec", 0.0)))
        hydra_recalls.append(float(room_metrics.get("hydra_recall", 0.0)))
        hydra_recalls_all.append(float(room_metrics.get("hydra_recall_all", 0.0)))

        # convert numpy types to native Python for JSON serialization
        serializable_metrics = {}
        for k, v in room_metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                serializable_metrics[k] = float(v)
            else:
                serializable_metrics[k] = v

        all_frame_metrics[frame_dir_name] = serializable_metrics

        print(f"Room metrics for {frame_dir_name}: {serializable_metrics}")

    # --- Save all metrics to a single JSON ---
    metrics_path = os.path.join(eval_output_dir, "room_metrics_per_frame.json")
    with open(metrics_path, "w") as f:
        json.dump(all_frame_metrics, f, indent=2)
    print(f"\nAll room metrics saved to {metrics_path}")

    # --- Plot hydra precision over frames ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame_indices, hydra_precisions, marker="o", linewidth=2, label="Hydra Precision")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Hydra Precision")
    ax.set_title(f"Hydra Precision over Frames — {params.main.scene_id}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    prec_plot_path = os.path.join(eval_output_dir, "hydra_precision_over_frames.png")
    fig.savefig(prec_plot_path, dpi=150)
    plt.close(fig)
    print(f"Hydra precision plot saved to {prec_plot_path}")

    # --- Plot hydra recall over frames ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame_indices, hydra_recalls, marker="o", linewidth=2, color="tab:orange", label="Hydra Recall (visible GT)")
    ax.plot(frame_indices, hydra_recalls_all, marker="x", linewidth=1.5, color="tab:orange", linestyle="--", alpha=0.6, label="Hydra Recall (all GT)")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Hydra Recall")
    ax.set_title(f"Hydra Recall over Frames — {params.main.scene_id}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    recall_plot_path = os.path.join(eval_output_dir, "hydra_recall_over_frames.png")
    fig.savefig(recall_plot_path, dpi=150)
    plt.close(fig)
    print(f"Hydra recall plot saved to {recall_plot_path}")

    # --- Combined plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame_indices, hydra_precisions, marker="o", linewidth=2, label="Hydra Precision")
    ax.plot(frame_indices, hydra_recalls, marker="s", linewidth=2, label="Hydra Recall (visible GT)")
    ax.plot(frame_indices, hydra_recalls_all, marker="x", linewidth=1.5, linestyle="--", alpha=0.6, label="Hydra Recall (all GT)")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Score")
    ax.set_title(f"Hydra Precision & Recall over Frames — {params.main.scene_id}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    combined_plot_path = os.path.join(eval_output_dir, "hydra_prec_recall_over_frames.png")
    fig.savefig(combined_plot_path, dpi=150)
    plt.close(fig)
    print(f"Combined plot saved to {combined_plot_path}")

    print(f"\nIncremental evaluation complete — {len(frame_indices)} frames evaluated.")


if __name__ == "__main__":
    main()

