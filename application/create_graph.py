import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hovsg.graph.graph import Graph

# pylint: disable=all


def process_scene(params: DictConfig):
    """Process a single scene: create feature map, save outputs, and optionally build graph."""
    save_dir = params.main.save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # create graph object (loads models and dataset once per scene)
    hovsg = Graph(params)

    incremental = getattr(params.pipeline, "incremental", False)

    if not incremental:
        # --- Original single-graph mode ---
        hovsg.create_feature_map()

        hovsg.save_masked_pcds(path=save_dir, state="both")
        hovsg.save_full_pcd(path=save_dir)
        hovsg.save_full_pcd_feats(path=save_dir)

        print(params.main.dataset)
        if params.main.dataset != "replica" and params.main.dataset != "scannet" and params.pipeline.create_graph:
            hovsg.build_graph(save_path=save_dir)
        else:
            print("Skipping hierarchical scene graph creation for Replica and ScanNet datasets.")
    else:
        # --- Incremental mode: build a new graph at every frame step ---
        skip = params.pipeline.skip_frames
        dataset_len = len(hovsg.dataset)
        # frame indices that will be processed: 10, 10+skip, 10+2*skip, ...
        frame_indices = list(range(10, dataset_len, skip))
        # Start from the third idx so the creation graph logic works
        # (problems with floors [at least has to be one] and rooms detections [at least has to be two o them])
        frame_indices = frame_indices[4:]
        if frame_indices[-1] < dataset_len - 1:
            frame_indices.append(dataset_len - 1)

        for step_idx, max_frame_idx in enumerate(frame_indices):

            step_label = f"frame_{max_frame_idx:06d}"
            step_dir = os.path.join(save_dir, step_label)
            os.makedirs(step_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Incremental step {step_idx + 1}/{len(frame_indices)} "
                  f"— processing frames 0..{max_frame_idx} "
                  f"(max_frame={max_frame_idx})")
            print(f"{'='*60}")

            # reset accumulated state from previous step
            hovsg.reset()

            # build feature map with only the frames seen so far
            hovsg.create_feature_map(max_frame=max_frame_idx)

            # save outputs for this step
            hovsg.save_masked_pcds(path=step_dir, state="both")
            hovsg.save_full_pcd(path=step_dir)
            hovsg.save_full_pcd_feats(path=step_dir)

            # build hierarchical graph if applicable
            print(params.main.dataset)
            if params.main.dataset != "replica" and params.main.dataset != "scannet" and params.pipeline.create_graph:
                hovsg.build_graph(save_path=step_dir, max_frame=max_frame_idx)
            else:
                print("Skipping hierarchical scene graph creation for Replica and ScanNet datasets.")

        print(f"\nIncremental graph creation complete — {len(frame_indices)} steps saved under {save_dir}")


@hydra.main(version_base=None, config_path="../config", config_name="create_graph")
def main(params: DictConfig):
    # discover all scene directories under dataset_path/split that don't start with "no_use"
    scenes_root = os.path.join(params.main.dataset_path, params.main.split)
    scene_ids = sorted([
        d for d in os.listdir(scenes_root)
        if os.path.isdir(os.path.join(scenes_root, d)) and not d.startswith("no_use")
    ])

    print(f"Found {len(scene_ids)} scenes in {scenes_root}:")
    for sid in scene_ids:
        print(f"  - {sid}")

    for scene_idx, scene_id in enumerate(scene_ids):
        print(f"\n{'#'*60}")
        print(f"Scene {scene_idx + 1}/{len(scene_ids)}: {scene_id}")
        print(f"{'#'*60}")

        # create a per-scene copy of the config with updated paths
        scene_params = OmegaConf.create(OmegaConf.to_container(params, resolve=True))
        scene_params.main.scene_id = scene_id
        scene_params.main.save_path = os.path.join(params.main.save_path, params.main.dataset, scene_id)
        scene_params.main.dataset_path = os.path.join(scenes_root, scene_id)

        process_scene(scene_params)

    print(f"\nAll {len(scene_ids)} scenes processed.")


if __name__ == "__main__":
    main()