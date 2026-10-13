"""Build the lab report and annotated figures from measured results only."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, _match, evaluate, parse_output


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fnum(value):
    return f"{value:.4f}"


def ci(result):
    return "[" + ", ".join(fnum(x) for x in result["f1_ci95"]) + "]"


def generate(name):
    from PIL import Image, ImageDraw
    meta = load("data/splits/meta.json")
    model = load("report/model_checks.json")
    evidence = {key: load(f"report/evidence/{key}.json") for key in ("e0", "e1", "e2", "e3")}
    # Earlier pipeline revisions scored E0 without resource metadata. Add only
    # measured metadata through the original scorer, retaining identical outputs.
    if "peak_vram_gb" not in evidence["e0"].get("meta", {}):
        e0_run = load("work/e0.run.json")
        subprocess.run([sys.executable, "scripts/score.py", "--pred", "work/e0.jsonl",
                        "--name", "E0", "--out", "report/evidence/e0.json",
                        "--meta", "trainable_params=0", "--meta", "train_minutes=0",
                        "--meta", f"peak_vram_gb={e0_run['peak_vram_gb']}"], check=True)
        evidence["e0"] = load("report/evidence/e0.json")
    e4_paths = sorted(Path("report/evidence").glob("e4*.json"))
    if len(e4_paths) != 2:
        raise ValueError("Both E4 data-size conditions must be scored before report generation")
    evidence.update({p.stem: load(p) for p in e4_paths})
    entries = read_jsonl("data/splits/test.jsonl")
    best = load("report/selection.json")["best_setup"]
    pred_rows = read_jsonl(f"work/{best.lower()}.jsonl")
    preds = {r["image_id"]: r["raw"] for r in pred_rows}
    folder = Path("report/figures")
    folder.mkdir(parents=True, exist_ok=True)
    candidates = []
    for entry in entries:
        parsed = parse_output(preds[entry["image_id"]])
        per, _ = _match(_gt_boxes(entry), parsed or [])
        counts = [sum(v[i] for v in per.values()) for i in range(3)]
        candidates.append((entry, parsed, counts))
    selected = []
    used = set()
    # Include success and failures, plus class-specific examples when available.
    groups = [lambda x: x[2][0] > 0 and x[2][1] == x[2][2] == 0,
              lambda x: x[2][1] > 0, lambda x: x[2][2] > 0]
    groups += [(lambda x, lab=lab: any(o["label"] == lab for o in x[0]["objects"])) for lab in CLASSES]
    for condition in groups:
        for candidate in candidates:
            if candidate[0]["image_id"] not in used and condition(candidate):
                selected.append(candidate)
                used.add(candidate[0]["image_id"])
                break
    for candidate in candidates:
        if len(selected) >= 6:
            break
        if candidate[0]["image_id"] not in used:
            selected.append(candidate)
            used.add(candidate[0]["image_id"])
    illustrations = []
    for i, (entry, parsed, counts) in enumerate(selected[:6], 1):
        image = Image.open(Path("data") / entry["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for boxes, color, prefix in ((_gt_boxes(entry), "lime", "GT"), (parsed or [], "red", "PRED")):
            for box in boxes:
                x1, y1, x2, y2 = box["bbox"]
                xy = (x1/1000*image.width, y1/1000*image.height, x2/1000*image.width, y2/1000*image.height)
                draw.rectangle(xy, outline=color, width=2)
                draw.text((max(0, xy[0]), max(0, xy[1]-12)), prefix+":"+box["label"], fill=color)
        draw.rectangle((0, 0, image.width, 18), fill="black")
        draw.text((4, 3), f"{best} | TP={counts[0]} FP={counts[1]} FN={counts[2]} | GT green / PRED red", fill="white")
        path = folder / f"prediction_{i}.jpg"
        image.save(path, quality=92)
        illustrations.append(f"![Ví dụ {i}]({path.as_posix()})\n\n"
            f"Ảnh `{entry['image_id']}`: TP={counts[0]}, FP={counts[1]}, FN={counts[2]}; "
            f"JSON {'hợp lệ' if parsed is not None else 'không hợp lệ'}. Box thật màu xanh, dự đoán màu đỏ.")
    area = {lab: [] for lab in CLASSES}
    for entry in entries:
        for obj in entry["objects"]:
            area[obj["label"]].append(obj["yolo"][2] * obj["yolo"][3] * 100)
    area_text = "; ".join(f"{lab}: {statistics.median(area[lab]):.3f}%" for lab in CLASSES)
    e0_raw = {r["image_id"]: r["raw"] for r in read_jsonl("work/e0.jsonl")}
    error_summary = {"invalid_json": 0, "images_with_fp": 0, "images_with_fn": 0,
                     "predictions_unknown_label": 0, "boxes_outside_0_1000": 0}
    examples = []
    for entry in entries:
        raw = e0_raw[entry["image_id"]]
        parsed = parse_output(raw)
        if parsed is None:
            error_summary["invalid_json"] += 1
        else:
            error_summary["predictions_unknown_label"] += sum(p["label"] not in CLASSES for p in parsed)
            error_summary["boxes_outside_0_1000"] += sum(any(v < 0 or v > 1000 for v in p["bbox"]) for p in parsed)
        per, _ = _match(_gt_boxes(entry), parsed or [])
        fp, fn = sum(v[1] for v in per.values()), sum(v[2] for v in per.values())
        error_summary["images_with_fp"] += fp > 0
        error_summary["images_with_fn"] += fn > 0
        if (fp or fn or parsed is None) and len(examples) < 2:
            examples.append(f"Ảnh `{entry['image_id']}`, FP={fp}, FN={fn}:\n\n```text\n{raw}\n```")
    Path("report/e0_error_analysis.json").write_text(json.dumps(error_summary, indent=2), encoding="utf-8")
    category_names = {"invalid_json": "JSON không parse được", "images_with_fp": "phát hiện thừa/sai (FP)",
                      "images_with_fn": "bỏ sót (FN)"}
    dominant = sorted(category_names, key=lambda key: error_summary[key], reverse=True)[:2]
    dominant_text = " và ".join(f"{category_names[key]} ({error_summary[key]} ảnh)" for key in dominant)
    e1_preds = {r["image_id"]: r["raw"] for r in read_jsonl("work/e1.jsonl")}
    wrong = evaluate(entries, e1_preds, coord="pixel")
    Path("report/coordinate_diagnostic.json").write_text(json.dumps(wrong, indent=2), encoding="utf-8")
    table = ["| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GiB) | Phút train | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |",
             "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    setup_labels = {"e0": "Zero-shot", "e1": "LoRA LLM", "e2": "+ projector", "e3": "+ LoRA ViT"}
    for key, result in evidence.items():
        m = result.get("meta", {})
        description = setup_labels.get(key, key.replace("e4_data_size_", "E4 train_fraction="))
        table.append(f"| {result['name']} | {description} | {int(m.get('trainable_params', 0)):,} | "
            f"{m.get('peak_vram_gb',0):.3f} | {m.get('train_minutes',0):.2f} | {fnum(result['micro']['f1'])} | "
            f"{ci(result)} | {result['fp_per_image']:.3f} | " + " | ".join(fnum(result["per_class"][lab]["f1"]) for lab in CLASSES) + " |")
    pp = model["parameters"]
    comparison = "\n".join(f"- {key.upper()}: F1={fnum(evidence[key]['micro']['f1'])}, CI95={ci(evidence[key])}, "
        f"VRAM={evidence[key]['meta']['peak_vram_gb']:.3f} GiB, train={evidence[key]['meta']['train_minutes']:.2f} phút."
        for key in ("e1", "e2", "e3"))
    counts = "\n".join(f"- {s}: {meta['counts'][s]['images']} ảnh; " + ", ".join(f"{c}={meta['counts'][s]['boxes'][c]} box" for c in CLASSES) for s in ("train", "val", "test"))
    worst = min(CLASSES, key=lambda c: evidence[best.lower()]["per_class"][c]["f1"])
    peak_key = max(("e1", "e2", "e3"), key=lambda k: evidence[k]["meta"]["peak_vram_gb"])
    delta_params = evidence["e2"]["meta"]["trainable_params"] - evidence["e1"]["meta"]["trainable_params"]
    e4_description = "\n".join(f"- {k}: F1={fnum(evidence[k]['micro']['f1'])}, CI95={ci(evidence[k])}." for k in evidence if k.startswith("e4"))
    widths = "; ".join(f"{k.upper()}: {evidence[k]['f1_ci95'][1]-evidence[k]['f1_ci95'][0]:.4f}" for k in ("e0", "e1", "e2", "e3"))
    selection = load("report/selection.json")
    report = f'''Họ tên: {name}
`dataset_version`: `{meta['dataset_version']}`

## Bảng tổng hợp

{chr(10).join(table)}

Các kết quả do `scripts/score.py` gốc tạo trên toàn bộ test; không sửa công cụ chấm. VRAM dùng `max_memory_allocated()/1024**3`, đơn vị GiB. Thời gian ở bảng là thời gian vòng train, gồm các checkpoint trước bước cuối, không gồm nạp model hay lưu checkpoint cuối; E0 không train. Thời gian suy luận được lưu riêng trong `work/*.run.json`.

## 10 câu hỏi

### Câu 1 — Kiến trúc và các khối được cập nhật

Đếm trên model chưa lượng tử hóa, khởi tạo ở meta device: LLM và phần embedding/head ngoài visual có {pp['llm']:,} tham số; backbone ViT có {pp['vit_backbone']:,}; projector có {pp['projector']:,}; tổng {pp['total']:,}. ViT trích đặc trưng ảnh; projector và các merger DeepStack đưa đặc trưng về không gian của LLM; LLM sinh JSON. E1 chỉ cập nhật LoRA LLM, E2 cập nhật thêm toàn bộ projector, E3 thêm LoRA backbone ViT. Số target đã xác nhận: {len(model['targets']['llm'])} LLM và {len(model['targets']['vit'])} ViT. E2 thêm {delta_params:,} tham số trainable so với E1.

### Câu 2 — Output và lỗi của E0

E0 có parse_valid_rate={evidence['e0']['parse_valid_rate']:.4f}. Kiểm tra từng ảnh ghi nhận {error_summary['invalid_json']} output không parse được, {error_summary['images_with_fp']} ảnh có FP và {error_summary['images_with_fn']} ảnh có FN. Hai biểu hiện nhiều nhất theo số ảnh là {dominant_text}. Các nhóm có thể trùng nhau: output không parse được được scorer coi là không có dự đoán và có thể tạo FN. Có {error_summary['predictions_unknown_label']} box dùng lớp ngoài danh sách và {error_summary['boxes_outside_0_1000']} box vượt phạm vi tọa độ. FP chưa đủ để phân biệt box bịa với box đúng vật thể nhưng lệch vị trí; cần xem ảnh minh họa.

{chr(10).join(examples)}

### Câu 3 — Kiểm chứng thang tọa độ

Giữ nguyên output E1, chấm đúng norm1000 có F1={fnum(evidence['e1']['micro']['f1'])}; cố ý diễn giải cùng output là pixel có F1={fnum(wrong['micro']['f1'])}. Kết quả kiểm chứng lưu ở `report/coordinate_diagnostic.json`. Cùng một số x=500 tương ứng 320 pixel trên ảnh rộng 640 khi dùng thang 1000; diễn giải thành 500 pixel làm lệch box và giảm IoU. Công thức ngược là x_pixel=x/1000*width, y_pixel=y/1000*height. Nhãn được đổi từ YOLO tâm/kích thước sang hai góc, làm tròn và kẹp [0,1000]; đã kiểm tra ví dụ [0.5,0.5,0.2,0.2] thành [400,400,600,600] và vẽ lại sáu ảnh nhãn.

### Câu 4 — Mask loss

Chỉ phần assistant được giám sát. Prompt, token ảnh và padding có label=-100. Kiểm tra tiền tố token của prompt khớp chính xác chuỗi đầy đủ; decode phần không mask ra JSON và token kết thúc. Một ảnh có {model['image_tokens']} token ảnh; mẫu decode được lưu ở `report/model_checks.json`. Nếu tính loss trên cả chuỗi, nhiều token prompt/ảnh cố định sẽ làm loãng mục tiêu học vị trí và lớp trong đáp án, đồng thời buộc model dự đoán các token đầu vào mà nhiệm vụ không yêu cầu. Không cắt/truncate chuỗi làm mất token kết thúc đáp án.

### Câu 5 — Lượng tử hóa và projector float32

QLoRA giữ base LLM ở NF4 4-bit và học ma trận LoRA. Các trọng số lượng tử hóa không phải tham số float thông thường để cập nhật toàn phần bằng AdamW; vì vậy cấu hình skip `model.visual`, `visual` và `lm_head`. Với Transformers 5.19.0, quy tắc khớp prefix khiến tên ngắn `visual` chưa đủ để loại trừ đường dẫn `model.visual`; mã đã kiểm tra nhánh visual không chứa Linear4bit. Ở E2/E3, projector được giữ ở float32; mọi tham số LoRA trainable cũng được chuyển sang float32 để GradScaler không gặp gradient fp16. Matmul vẫn đi qua autocast. Float32 là lựa chọn ổn định số của lab, không phải quy luật rằng mọi projector chỉ có thể train ở float32. Checkpoint lưu adapter và `projector.pt` riêng; inference kiểm tra và nạp lại cả hai cho E2/E3.

### Câu 6 — Chia theo phiên quay

Frame cùng phiên gần nhau có thể rất giống. Chia ảnh ngẫu nhiên đưa các frame gần trùng vào train và test khiến F1 đánh giá quá lạc quan khả năng tổng quát hóa. Giữ nguyên split do công cụ đề tạo, kiểm tra không có session giao nhau giữa mọi cặp split. Số liệu thực tế:

{counts}

Phân bố lớp giữa các phiên không đều, đặc biệt val chỉ có {meta['counts']['val']['boxes']['pothole']} box pothole, nên cần đọc F1 theo từng lớp.

### Câu 7 — So sánh E1/E2/E3 và tài nguyên

{comparison}

Setup cơ sở E4 là {best}, chọn bằng micro-F1 trên {selection['validation_images']} ảnh validation cố định, không chọn bằng điểm test. Các điểm validation lưu trong `report/selection.json`. VRAM đỉnh đo được lớn nhất là {peak_key.upper()}; mở projector thêm gradient và trạng thái optimizer, còn E3 phải backpropagate qua ViT để học LoRA. Không thể suy ra lợi ích chỉ từ số tham số: phải đối chiếu F1, FP/ảnh, F1 từng lớp và chi phí thực tế trong bảng. CI95 của từng setup là khoảng cho từng điểm số; khoảng chồng lấn không tự chứng minh hai setup tương đương, và khoảng không chồng lấn cũng không thay thế kiểm định chênh lệch ghép cặp.

### Câu 8 — Khoảng tin cậy và giới hạn kết luận

Độ rộng CI95 F1: {widths}. Công cụ chấm bootstrap 1000 lần theo ảnh. Test có {len(entries)} ảnh, ít mẫu cho một số lớp và các ảnh trong cùng phiên có tương quan; bootstrap theo ảnh có thể đánh giá thấp bất định giữa phiên. Vì vậy các chênh lệch nhỏ chỉ là tín hiệu trên split này, chưa chứng minh ưu thế ổn định trên tuyến đường hoặc phiên quay khác. Không tối ưu prompt hay chọn checkpoint bằng test.

### Câu 9 — Phân tích theo lớp và hình minh họa

Trong {best}, lớp thấp nhất là {worst}, F1={fnum(evidence[best.lower()]['per_class'][worst]['f1'])}. Diện tích box trung vị trên test theo phần trăm ảnh: {area_text}. Box nhỏ nhạy với lệch vài pixel; số mẫu và sự giống nhau về hình dạng có thể góp phần gây lỗi. Đây là các giả thuyết, không phải kết luận nhân quả từ một lần chạy. Sáu ví dụ sau hiển thị box thật và dự đoán, kèm TP/FP/FN theo chính quy tắc ghép của scorer.

{chr(10).join(illustrations)}

### Câu 10 — Ablation E4

Thay duy nhất tỷ lệ ảnh train: 0.1 và 1.0, giữ setup {best}, rank, alpha, learning rate, batch, gradient accumulation, seed và số optimizer steps. Tập 10% là mẫu cố định theo seed 42; giá trị 100% dùng lại run cơ sở đúng cấu hình và chấm dưới tên E4. Cả hai đều 300 steps nên 10% được lặp lại nhiều hơn; phép thử đánh giá lượng dữ liệu độc nhất dưới cùng ngân sách cập nhật, không phải cùng số epoch.

{e4_description}

Diễn giải kết quả với CI95 và các giới hạn của test nhỏ ở câu 8; không khẳng định ý nghĩa thống kê chỉ từ hai khoảng riêng lẻ.

## Nhìn lại

Mã triển khai có kiểm tra prefix chat template, mask đáp án, module LoRA, loss/gradient hữu hạn và nạp lại projector. Scorer có thể parse JSON hợp lệ nhưng chưa chứng minh box đúng; cần kết hợp F1, FP/FN và ảnh. Chưa thực hiện E5 GRPO tùy chọn. Các kết quả phản ánh những lần chạy được ghi trong evidence và train logs, không thay thế việc tự xem lại mã và giải thích trong phần vấn đáp.
'''
    Path("REPORT.md").write_text(report, encoding="utf-8")
    print("REPORT.md generated from measured evidence; six prediction figures saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Nguyễn Văn Tứ")
    generate(parser.parse_args().name)
