"""PyCharm 直接 Run：静态检查、构建真实单类(target)S/X，并检查当前配置的数据；不训练。"""
import ast
import time
import traceback
import uuid

from config import CLASS_NAMES, CONFIG, MODEL_FILES, MODEL_ORDER, PROJECT_ROOT
from tinywod.common import write_json


def main():
    from tinywod.runner import validate_config
    folder = PROJECT_ROOT / "verification" / ("check_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    report = {"static": False, "models": [], "dataset": "not_checked", "real_training": False}
    try:
        paths = [*PROJECT_ROOT.glob("*.py"), *PROJECT_ROOT.joinpath("tinywod").glob("*.py"),
                 *PROJECT_ROOT.joinpath("ultralytics").rglob("*.py")]
        for p in paths:
            ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
        validate_config(CONFIG, MODEL_ORDER, formal=True)
        assert MODEL_ORDER == ("s", "x") and CONFIG.epochs == 200
        report.update(static=True, python_files=len(paths))
        from tinywod.worker import environment, choose_model, structure
        environment(CONFIG, folder)
        from ultralytics.nn.tasks import DetectionModel
        for variant in MODEL_ORDER:
            _, record = choose_model(CONFIG, variant, folder / variant)
            model = DetectionModel(str(MODEL_FILES[variant]), nc=len(CLASS_NAMES), verbose=False)
            model.names = dict(enumerate(CLASS_NAMES))
            model.end2end = True
            spec = structure(model)
            report["models"].append(spec)
            print(f"[检查] YOLO26-{variant.upper()} 实际单类(target)参数={spec['parameters']:,} 结构指纹={spec['signature']}", flush=True)
            del model
        assert len({m["signature"] for m in report["models"]}) == 2
        from tinywod.dataset import prepare_dataset
        report["dataset"] = prepare_dataset(CONFIG, folder / "dataset_inputs")
        print("[检查] 全部通过。未执行训练。", flush=True)
        code = 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
        print("[检查] 失败项见报告；不会把缺少的数据或未训练标为通过。", flush=True)
        code = 1
    write_json(folder / "check_report.json", report)
    print(f"[检查报告] {folder / 'check_report.json'}", flush=True)
    return code


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    raise SystemExit(main())
