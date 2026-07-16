import sys
import os
import subprocess
import json

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *
from mea.paired import PROTOCOL_ID, load_seed_manifest

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base in place."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def class_decorator(task_name: str, task_module: str | None = None):
    module_name = task_module or f"envs.{task_name}"
    try:
        envs_module = importlib.import_module(module_name)
        env_class = getattr(envs_module, task_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"Task class {task_name!r} was not found in module {module_name!r}"
        ) from exc

    return env_class()


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    task_overlay = usr_args.get("task_overlay")
    if task_overlay:
        overlay_path = Path(task_overlay).expanduser().resolve()
        with open(overlay_path, "r", encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}

        if not isinstance(overlay, dict):
            raise ValueError(
                f"task_overlay must contain a YAML mapping, got {type(overlay).__name__}"
            )

        deep_update(args, overlay)
        print(f"Loaded task overlay from {overlay_path}")

    # Canonical identity always comes from the command line, not the overlay.
    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    requested_output_dir = usr_args.get("output_dir")
    save_dir = (
        Path(requested_output_dir).expanduser().resolve()
        if requested_output_dir
        else Path(
            f"eval_result/{task_name}/{policy_name}/{task_config}/"
            f"{ckpt_setting}/{current_time}"
        )
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(
        task_name=args["task_name"],
        task_module=usr_args.get("task_module"),
    )
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = int(usr_args["seed"])

    exact_manifest = None
    exact_seeds = None
    if usr_args.get("seed_manifest"):
        exact_manifest = load_seed_manifest(
            usr_args["seed_manifest"],
            expected_task_name=task_name,
        )
        matching_conditions = [
            condition
            for condition in exact_manifest["conditions"]
            if condition["task_config"] == task_config
        ]
        if len(matching_conditions) != 1:
            raise ValueError(
                f"task_config {task_config!r} is not a unique condition in "
                "the exact-seed manifest"
            )
        exact_seeds = list(exact_manifest["seeds"])

    st_seed = int(
        exact_seeds[0]
        if exact_seeds is not None
        else usr_args.get("start_seed", 100000 * (1 + seed))
    )
    suc_nums = []
    test_num = int(
        len(exact_seeds)
        if exact_seeds is not None
        else usr_args.get("num_episodes", 100)
    )
    if test_num <= 0:
        raise ValueError(f"num_episodes must be positive, got {test_num}")
    if exact_seeds is not None and usr_args.get("num_episodes") is not None:
        configured_count = int(usr_args["num_episodes"])
        if configured_count != test_num:
            raise ValueError(
                "num_episodes must equal the exact seed count: "
                f"{configured_count} != {test_num}"
            )

    print(f"Evaluation episodes: {test_num}")
    if exact_seeds is None:
        print(f"Evaluation start seed: {st_seed}")
    else:
        print(f"Exact evaluation seeds: {exact_seeds}")
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num, seed_measurements = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        telemetry_dir=usr_args.get("telemetry_dir"),
        telemetry_profile=usr_args.get("telemetry_profile", "balanced_v1"),
        task_module=usr_args.get("task_module"),
        exact_seeds=exact_seeds,
        return_measurements=True,
    )
    suc_nums.append(suc_num)

    if exact_manifest is not None:
        eligible_count = sum(
            row["eligibility_status"] == "passed"
            for row in seed_measurements
        )
        evaluated_count = sum(
            row["policy_executed"] for row in seed_measurements
        )
        seed_result = {
            "schema_version": 1,
            "protocol": PROTOCOL_ID,
            "task_name": task_name,
            "task_config": task_config,
            "condition_id": matching_conditions[0]["id"],
            "requested_seeds": exact_seeds,
            "requested_count": len(exact_seeds),
            "eligible_count": eligible_count,
            "evaluated_count": evaluated_count,
            "success_count": suc_num,
            "success_rate_evaluated": (
                suc_num / evaluated_count if evaluated_count else None
            ),
            "all_eligible": eligible_count == len(exact_seeds),
            "no_seed_replacement": [
                row["seed"] for row in seed_measurements
            ] == exact_seeds,
            "seed_measurements": seed_measurements,
        }
        seed_results_path = Path(
            usr_args.get("seed_results_path") or save_dir / "seed_results.json"
        ).expanduser().resolve()
        seed_results_path.parent.mkdir(parents=True, exist_ok=True)
        seed_results_path.write_text(
            json.dumps(seed_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Exact-seed results: {seed_results_path}")

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        denominator = (
            sum(row["policy_executed"] for row in seed_measurements)
            if exact_manifest is not None
            else test_num
        )
        rates = [
            value / denominator if denominator else float("nan")
            for value in suc_nums
        ]
        file.write("\n".join(map(str, rates)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                telemetry_dir=None,
                telemetry_profile="balanced_v1",
                task_module=None,
                exact_seeds=None,
                return_measurements=False):
    """Evaluate ACT in legacy scan mode or strict exact-seed mode.

    Legacy callers retain the upstream behavior: expert-ineligible seeds are
    skipped and the evaluator scans forward until ``test_num`` eligible scenes
    have run.  Supplying ``exact_seeds`` changes the contract: each requested
    seed is considered exactly once, in order, and every rejection/error is
    recorded instead of being replaced by a later seed.
    """

    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    seed_measurements = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    telemetry_root = (
        Path(telemetry_dir).expanduser().resolve()
        if telemetry_dir
        else None
    )

    strict_mode = exact_seeds is not None
    exact_seed_list = list(exact_seeds or [])
    exact_index = 0

    def close_env_safely(*, clear_cache=False):
        try:
            TASK_ENV.close_env(clear_cache=clear_cache)
        except Exception:
            traceback.print_exc()

    while (
        exact_index < len(exact_seed_list)
        if strict_mode
        else succ_seed < test_num
    ):
        requested_index = None
        measurement = None
        if strict_mode:
            requested_index = exact_index
            now_seed = exact_seed_list[exact_index]
            exact_index += 1
            measurement = {
                "requested_index": requested_index,
                "seed": now_seed,
                "eligibility_status": None,
                "policy_executed": False,
                "policy_success": None,
                "policy_status": "not_run",
                "time_to_success": None,
            }

        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                close_env_safely()
                args["render_freq"] = render_freq
                if strict_mode:
                    measurement.update(
                        {
                            "eligibility_status": "unstable",
                            "eligibility_error": {
                                "type": type(e).__name__,
                                "message": str(e),
                            },
                        }
                    )
                    seed_measurements.append(measurement)
                else:
                    now_seed += 1
                continue
            except Exception as e:
                close_env_safely()
                args["render_freq"] = render_freq
                print("error occurs !")
                if strict_mode:
                    measurement.update(
                        {
                            "eligibility_status": "error",
                            "eligibility_error": {
                                "type": type(e).__name__,
                                "message": str(e),
                            },
                        }
                    )
                    seed_measurements.append(measurement)
                else:
                    now_seed += 1
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
            if strict_mode:
                measurement["eligibility_status"] = "passed"
        else:
            args["render_freq"] = render_freq
            if strict_mode:
                measurement["eligibility_status"] = "expert_failed"
                seed_measurements.append(measurement)
                close_env_safely()
            else:
                now_seed += 1
            continue

        args["render_freq"] = render_freq
        try:
            TASK_ENV.setup_demo(
                now_ep_num=now_id,
                seed=now_seed,
                is_test=True,
                **args,
            )
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(
                args["task_name"], episode_info_list, test_num
            )
            instruction = np.random.choice(results[0][instruction_type])
            TASK_ENV.set_instruction(instruction=instruction)

            if TASK_ENV.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        video_size,
                        "-framerate",
                        "10",
                        "-i",
                        "-",
                        "-pix_fmt",
                        "yuv420p",
                        "-vcodec",
                        "libx264",
                        "-crf",
                        "23",
                        f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

            recorder = None
            episode_dir = None
            if telemetry_root is not None:
                # Import only for opt-in telemetry runs so the upstream
                # evaluator keeps the same dependency path by default.
                from mea.toolkit import EpisodeRecorder

                episode_dir = (
                    telemetry_root / f"episode_{now_id:03d}_seed_{now_seed}"
                )
                recorder = EpisodeRecorder(
                    Path.cwd(),
                    episode_dir,
                    task_name=args["task_name"],
                    seed=now_seed,
                    episode_index=now_id,
                    policy_name=policy_name,
                    task_module=task_module,
                    task_config=args.get("task_config"),
                    checkpoint_setting=args.get("ckpt_setting"),
                    telemetry_profile_id=str(telemetry_profile),
                )
                TASK_ENV._mea_recorder = recorder
                try:
                    recorder.start(TASK_ENV)
                except Exception:
                    TASK_ENV._mea_recorder = None
                    raise
                print(f"MEA telemetry: {episode_dir}")

            succ = False
            rollout_error = None
            try:
                reset_func(model)
                while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                    observation = TASK_ENV.get_obs()
                    eval_func(TASK_ENV, model, observation)
                    if TASK_ENV.eval_success:
                        succ = True
                        break
            except BaseException as exc:
                rollout_error = exc
                if recorder is not None:
                    recorder.record_error(exc)
                raise
            finally:
                if TASK_ENV.eval_video_path is not None:
                    TASK_ENV._del_eval_video_ffmpeg()
                if recorder is not None:
                    TASK_ENV._mea_recorder = None
                    error_payload = (
                        {
                            "type": type(rollout_error).__name__,
                            "message": str(rollout_error),
                        }
                        if rollout_error is not None
                        else None
                    )
                    try:
                        recorder.finish(
                            TASK_ENV,
                            success=succ or bool(TASK_ENV.eval_success),
                            error=error_payload,
                        )
                    except Exception:
                        if rollout_error is None:
                            raise
                        traceback.print_exc()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            if not strict_mode:
                raise
            measurement.update(
                {
                    "policy_status": "error",
                    "policy_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "execution_attempted": True,
                }
            )
            seed_measurements.append(measurement)
            close_env_safely(clear_cache=True)
            continue

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        if strict_mode:
            measurement.update(
                {
                    "policy_executed": True,
                    "policy_success": bool(succ),
                    "policy_status": "success" if succ else "failure",
                    "execution_attempted": True,
                    "telemetry_episode_dir": (
                        str(episode_dir) if episode_dir is not None else None
                    ),
                }
            )
            seed_measurements.append(measurement)

        now_id += 1
        TASK_ENV.close_env(
            clear_cache=((succ_seed + 1) % clear_cache_freq == 0)
        )

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        if not strict_mode:
            now_seed += 1

    if return_measurements:
        return now_seed, TASK_ENV.suc, seed_measurements
    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
