"""Build a readable, self-contained Colab notebook without credentials or weights."""
import json
from pathlib import Path


def main():
    cells = []
    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.splitlines(keepends=True)})
    md("# Lab Day 10 — Qwen3-VL-4B, QLoRA và định vị hư hại mặt đường\n"
       "Triển khai đầy đủ E0–E4. Kết quả được chấm bằng công cụ gốc; không điền số liệu trước khi chạy. "
       "Mở runtime T4. Mỗi run khởi tạo lại model gốc; adapter và projector lưu ngoài Git.\n")
    md("## 0. Môi trường (Colab T4)\nNotebook ghi mã thực hiện vào runtime nên có thể chạy lại độc lập. "
       "Test dành riêng cho REPORT còn placeholder được bỏ qua khi kiểm tra repo đã điền báo cáo; giữ nguyên file test.")
    code("import os, sys, subprocess\nfrom pathlib import Path\n"
         "REPO = Path('/content/road_damage_lab')\n"
         "if not (REPO / 'README.md').exists():\n"
         "    subprocess.run(['git', 'clone', 'https://github.com/nguyenvantu19/K4-Track-4-Day10-VisionLanguageModel.git', str(REPO)], check=True)\n"
         "os.chdir(REPO)\n!nvidia-smi -L\n!pip -q install -r requirements-colab.txt pytest\n"
         "# torchao không dùng trong lab bitsandbytes; bản có sẵn ở Colab xung đột PEFT.\n"
         "!pip -q uninstall -y torchao\n"
         "!python -m pytest -q -k 'not test_report_template_has_student_markers_and_ten_questions'\n")
    md("## 1. Dữ liệu: tải và chia split theo phiên\n"
       "Dùng ZIP công khai của dataset đúng đề. Công cụ prepare_data giữ nguyên, không tự chia lại split.")
    code("if not Path('data/splits/meta.json').exists():\n"
         "    subprocess.run([sys.executable, 'scripts/prepare_data.py', '--zip', 'https://www.kaggle.com/api/v1/datasets/download/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes'], check=True)\n"
         "import json\nmeta = json.loads(Path('data/splits/meta.json').read_text())\n"
         "print(meta['dataset_version'])\nprint(json.dumps(meta['counts'], indent=2))\n")
    md("## 2. Nhãn YOLO → JSON thang 0–1000\n"
       "Cell sau chứa mã triển khai: chuyển nhãn, encode và mask, LoRA, train, lưu/nạp projector, inference và kiểm tra kiến trúc. "
       "`yolo_to_box` và `objects_to_answer` được dùng thống nhất cho dữ liệu và ảnh kiểm chứng.")
    code("%%writefile student_lab.py\n" + Path("student_lab.py").read_text(encoding="utf-8"))
    code("!python student_lab.py inspect-data\n"
         "from IPython.display import display\nfrom PIL import Image\n"
         "display(Image.open('report/figures/ground_truth_1.jpg'))\n")
    md("## 3. E0: zero-shot\n"
       "Model NF4, bỏ lượng tử hóa visual/lm_head, padding trái, decode riêng phần sinh thêm. "
       "Pipeline ở bước 8 chạy đủ test và chấm E0. Có thể dùng cell này để kiểm tra hai ảnh val trước.")
    code("!python -u student_lab.py infer --name E0_debug --split val --limit 2 --batch-size 2 --output work/e0_debug.jsonl\n"
         "print(Path('work/e0_debug.jsonl').read_text())\n")
    md("## 4. Chat template và mask loss\n"
       "Kiểm tra prompt là tiền tố của full input. Labels prompt/padding=-100; pixel_values và image_grid_thw được nối, không pad. "
       "Decode phần labels không mask để kiểm tra JSON và token kết thúc.")
    code("!python student_lab.py verify-model\n"
         "checks = json.loads(Path('report/model_checks.json').read_text())\n"
         "print('Image tokens:', checks['image_tokens'])\nprint('Answer-only labels:', checks['decoded_labels'])\n"
         "print('Parameter counts:', checks['parameters'])\n")
    md("## 5. Setup LoRA E1/E2/E3\n"
       "E1: LoRA LLM; E2: thêm projector toàn phần float32; E3: thêm LoRA ViT. "
       "Target module được kiểm tra bằng tên đầy đủ: 252 LLM, 96 ViT. Rank 16, alpha 32, dropout 0.05.")
    code("print('LLM targets:', len(checks['targets']['llm']))\n"
         "print('ViT targets:', len(checks['targets']['vit']))\n"
         "assert len(checks['targets']['llm']) == 252\nassert len(checks['targets']['vit']) == 96\n")
    md("## 6. Vòng huấn luyện và checkpoint\n"
       "AdamW, warmup 5%, cosine, 300 optimizer steps, gradient checkpointing, fp16 GradScaler và clip norm 1. "
       "Mỗi 100 steps lưu adapter, projector riêng, optimizer/scheduler/scaler và metadata. "
       "Overflow gradient sẽ giảm loss scale và thử lại, không được tính thành optimizer step. "
       "Có thể lưu checkpoint vào Drive nếu Drive đã được mount; mặc định dùng work/checkpoints trong runtime.")
    code("%%writefile make_report.py\n" + Path("make_report.py").read_text(encoding="utf-8"))
    code("%%writefile run_lab.py\n" + Path("run_lab.py").read_text(encoding="utf-8"))
    code("%%writefile colab_runner.py\n" + Path("colab_runner.py").read_text(encoding="utf-8"))
    md("## 7. Suy luận và chấm E1–E3\n"
       "Nạp base model mới, adapter và projector tương ứng; chạy test đủ ảnh, ghi raw output, "
       "chấm bằng scripts/score.py với số tham số, VRAM và thời gian thực tế. "
       "40 ảnh đầu của val cố định dùng chọn setup cơ sở E4; báo cáo ghi rõ đây là subset.")
    code("CHECKPOINT_ROOT = 'work/checkpoints'\n"
         "print('Checkpoint directory:', CHECKPOINT_ROOT)\n")
    md("## 8. E4: ablation dữ liệu và chạy toàn bộ pipeline\n"
       "Chọn cơ sở bằng validation F1, so sánh 10% và 100% ảnh train, giữ các siêu tham số khác như nhau. "
       "Run 100% được dùng lại; run 10% bắt đầu từ model gốc. E5 GRPO là phần tùy chọn chưa triển khai. "
       "Nếu dừng giữa chừng, chạy lại pipeline để tiếp tục từ checkpoint và các dự đoán đã ghi.")
    code("from colab_runner import start_colab\n"
         "if 'lab_thread' in globals() and lab_thread.is_alive():\n"
         "    raise RuntimeError('Pipeline đang chạy; dùng ô theo dõi bên dưới.')\n"
         "lab_thread = start_colab(['--checkpoints', CHECKPOINT_ROOT])\n"
         "print('Pipeline đang chạy nền. Chờ stage=complete ở ô kiểm tra tiếp theo.')\n")
    code("from colab_runner import poll_colab\n"
         "state = poll_colab()\n"
         "print('Chạy lại ô này để xem tiến độ và tải checkpoint mới.')\n")
    md("## 9. Báo cáo và xuất bài\n"
       "REPORT được tạo sau khi đủ evidence. Sáu ảnh có GT xanh và prediction đỏ; "
       "toàn bộ nhận xét lấy số từ các run thật. Check_submission cần ít nhất 5 commit trong repo nộp; "
       "commit từ các giai đoạn phát triển trên máy cá nhân và kiểm tra lại trước khi đẩy GitHub.")
    code("assert json.loads(Path('work/pipeline_status.json').read_text())['stage'] == 'complete', 'Chờ pipeline hoàn thành'\n"
         "from IPython.display import Markdown\ndisplay(Markdown(Path('REPORT.md').read_text()))\n"
         "!python scripts/check_submission.py\n")
    code("from google.colab import files\nfiles.download('work/lab_submission.zip')\n")
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
    Path("lab_day10.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
