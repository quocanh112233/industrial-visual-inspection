"""FR-15 — Sinh bang so sanh, bieu do va bao cao tu results/benchmark.json.

Moi con so trong bao cao deu doc tu file JSON, khong go tay (NFR-06). Neu
benchmark chua chay du, bao cao ghi ro o nao con thieu thay vi bo trong.

    PYTHONPATH=src python -m ivid.benchmark.report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..data.common import repo_root

BACKEND_LABEL = {"pytorch": "PyTorch", "onnx": "ONNX Runtime", "tensorrt": "TensorRT FP16"}
BACKEND_ORDER = ["pytorch", "onnx", "tensorrt"]


def rows_from(payload: dict) -> list[dict]:
    """Gop latency + accuracy + tai nguyen thanh mot dong moi (model, backend)."""
    runs, acc = payload.get("runs", {}), payload.get("accuracy", {})
    memory = payload.get("memory", {})
    keys = sorted(set(runs) | set(acc),
                  key=lambda k: (k.split("|")[0],
                                 BACKEND_ORDER.index(k.split("|")[1])
                                 if k.split("|")[1] in BACKEND_ORDER else 99))
    out = []
    for k in keys:
        model, backend = k.split("|", 1)
        r, an, mem = runs.get(k, {}), acc.get(k, {}), memory.get(k, {})
        row = {"model": model, "backend": backend,
               "backend_label": BACKEND_LABEL.get(backend, backend)}
        if r and not r.get("skipped"):
            ms = r["median_stages"]
            row.update(
                p50=r["latency_p50_ms"], p95=r["latency_p95_ms"], fps=r["fps"],
                pre=ms["preprocess"]["p50"], infer=ms["inference"]["p50"],
                post=ms["postprocess"]["p50"],
                size_mb=r["model_size_mb"],
                gpu_mb=r["resources"].get("torch_gpu_peak_mb"),
                rss_mb=r["resources"].get("process_rss_peak_mb"),
                sys_delta_mb=r["resources"].get("system_used_delta_mb"),
                spread=r.get("session_p50_spread_percent"),
                on_gpu=r["backend_info"].get("running_on_gpu", True),
                warn=r["backend_info"].get("CANH_BAO"),
            )
        else:
            row["latency_missing"] = (r or {}).get("reason", "chua do")
        if mem and not mem.get("skipped"):
            row["mem_delta_mb"] = mem.get("rss_delta_mb")
            row["load_s"] = mem.get("nap_giay")
        if an and not an.get("skipped"):
            # accuracy.py moi tra ve mAP o cap cao nhat (khong con boc trong "overall"),
            # va per_class danh so theo CHI SO lop. Doi sang ten lop de bang doc duoc.
            names = an.get("class_names") or []
            pc = {}
            for cid, v in (an.get("per_class") or {}).items():
                label = names[int(cid)] if int(cid) < len(names) else str(cid)
                pc[label] = v
            row.update(map50=an["mAP50"], map5095=an["mAP50_95"],
                       precision=an["precision"], recall=an["recall"], per_class=pc)
        else:
            row["accuracy_missing"] = (an or {}).get("reason", "chua do")
        out.append(row)
    return out


def fmt(v, nd=2, dash="—"):
    if v is None:
        return dash
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def make_charts(rows: list[dict], out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] chua cai matplotlib -> bo qua bieu do")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    have = [r for r in rows if "p50" in r]
    if not have:
        return []

    # --- 1. cot do tre theo dinh dang ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"{r['model']}\n{r['backend_label']}" for r in have]
    x = range(len(have))
    ax.bar([i - 0.2 for i in x], [r["p50"] for r in have], width=0.4, label="p50")
    ax.bar([i + 0.2 for i in x], [r["p95"] for r in have], width=0.4, label="p95")
    for i, r in enumerate(have):
        ax.text(i - 0.2, r["p50"], f"{r['p50']:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Độ trễ end-to-end (ms)")
    ax.set_title("Độ trễ theo định dạng runtime — Jetson Orin Nano")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "latency_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    # --- 2. thanh phan pre/infer/post ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottoms = [0.0] * len(have)
    for stage, key in (("Tiền xử lý", "pre"), ("Inference", "infer"), ("Hậu xử lý (NMS)", "post")):
        vals = [r.get(key, 0) for r in have]
        ax.bar(labels, vals, bottom=bottoms, label=stage)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    ax.set_ylabel("ms (p50)")
    ax.set_title("Thời gian đi đâu: tiền xử lý / inference / hậu xử lý")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "latency_breakdown.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    # --- 3. mAP vs do tre ---
    pts = [r for r in have if "map50" in r]
    if pts:
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in pts:
            ax.scatter(r["p50"], r["map50"], s=90)
            ax.annotate(f"{r['model']} {r['backend_label']}", (r["p50"], r["map50"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8)
        ax.set_xlabel("Độ trễ p50 (ms) — càng trái càng nhanh")
        ax.set_ylabel("mAP@0.5 — càng cao càng chính xác")
        ax.set_title("Đánh đổi tốc độ ↔ độ chính xác")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / "map_vs_latency.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p.name)

    return made


def main_table(rows: list[dict]) -> list[str]:
    # Cot bo nho lay tu `ivid.benchmark.memprobe`, KHONG lay tu phep do chay kem
    # benchmark. Hai cach truoc deu sai theo hai kieu khac nhau:
    #   * torch.cuda.max_memory_allocated(): bao 0 MB cho ONNX (ORT tu cap phat)
    #     va chi dem buffer vao/ra cho TensorRT -> nguoi doc se ket luan "ONNX
    #     khong ton bo nho GPU", sai hoan toan.
    #   * system_used_delta_mb: dem ca may, ke ca tien trinh khac va bo dem trang
    #     -> hai lan chay cung cau hinh ra 31.2 MB va 0.0 MB. Do la nhieu.
    # memprobe chay moi runtime trong mot tien trinh RIENG va lay RSS dinh diem
    # tru RSS luc khoi dong, nen con so so sanh duoc giua ba runtime.
    L = ["| Model | Runtime | mAP@0.5 | mAP@0.5:0.95 | p50 (ms) | p95 (ms) | FPS | "
         "Model size | RAM tiến trình | Nạp (s) |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        ram = (f"{fmt(r.get('mem_delta_mb'), 0)} MB" if r.get("mem_delta_mb") is not None
               else "— *(`make mem`)*")
        L.append(
            f"| {r['model']} | {r['backend_label']} | {fmt(r.get('map50'), 4)} | "
            f"{fmt(r.get('map5095'), 4)} | {fmt(r.get('p50'))} | {fmt(r.get('p95'))} | "
            f"{fmt(r.get('fps'), 1)} | {fmt(r.get('size_mb'), 1)} MB | {ram} | "
            f"{fmt(r.get('load_s'), 1) if r.get('load_s') is not None else '—'} |")
    return L


def conclusions(rows: list[dict], cycle_ms: float) -> list[str]:
    L: list[str] = []
    by = {(r["model"], r["backend"]): r for r in rows}
    models = sorted({r["model"] for r in rows})

    # --- Chi phi khoi dong: chi hien khi co chenh lech lon that su ---
    canh_bao_nap = []
    for m in models:
        trt, onx = by.get((m, "tensorrt")), by.get((m, "onnx"))
        if not (trt and onx and trt.get("load_s") and onx.get("load_s")):
            continue
        if onx["load_s"] >= 10 * trt["load_s"]:
            canh_bao_nap.append(
                f"- **{m}**: ONNX Runtime mất **{onx['load_s']:.0f} s** để nạp, "
                f"TensorRT chỉ **{trt['load_s']:.1f} s** ({onx['load_s'] / trt['load_s']:.0f}× lâu hơn), "
                f"và tốn **{onx['mem_delta_mb']:.0f} MB** RAM so với "
                f"**{trt['mem_delta_mb']:.0f} MB**")
    if canh_bao_nap:
        L.append("### 0. Chi phí khởi động — chỗ bảng độ trễ không nhìn thấy\n")
        L.extend(canh_bao_nap)
        L.append("")
        L.append(
            "ONNX Runtime ở đây chạy `TensorrtExecutionProvider`, nghĩa là nó **tự dựng "
            "engine TensorRT ngay lúc nạp model** — và dựng lại từ đầu mỗi lần tiến trình "
            "khởi động, vì bộ nhớ đệm engine chưa được bật. Trên dây chuyền, mỗi lần khởi "
            "động lại dịch vụ (mất điện, cập nhật, container bị lên lịch lại) là ngần ấy "
            "giây không kiểm được sản phẩm.\n")
        L.append(
            "Đây là lý do chọn định dạng không thể chỉ nhìn cột p50: ba runtime có độ trễ "
            "cùng bậc, nhưng chi phí khởi động lệch nhau hai bậc. Muốn dùng ONNX Runtime "
            "thì phải bật `trt_engine_cache_enable` và nung sẵn bộ nhớ đệm lúc đóng gói "
            "image; còn engine `.plan` dựng sẵn thì nạp thẳng trong 0.3 s.\n")

    # --- Cau 1 ---
    L.append("### 1. TensorRT FP16 nhanh hơn PyTorch bao nhiêu, đổi lấy bao nhiêu mAP?\n")
    any1 = False
    for m in models:
        pt, trt = by.get((m, "pytorch")), by.get((m, "tensorrt"))
        if not (pt and trt and "p50" in pt and "p50" in trt):
            continue
        any1 = True
        sp = pt["p50"] / trt["p50"]
        line = f"- **{m}**: nhanh hơn **{sp:.2f}×** ({pt['p50']:.2f} ms → {trt['p50']:.2f} ms)"
        if "map50" in pt and "map50" in trt:
            d = trt["map50"] - pt["map50"]
            pct = d / pt["map50"] * 100 if pt["map50"] else 0
            verdict = ("gần như không mất độ chính xác" if abs(d) < 0.005
                       else "mất độ chính xác đáng kể" if d < -0.02
                       else "mất một chút độ chính xác")
            line += (f", mAP@0.5 {pt['map50']:.4f} → {trt['map50']:.4f} "
                     f"(**{d:+.4f}**, {pct:+.1f}%) — {verdict}")
        else:
            line += ", *(chưa có số mAP)*"
        L.append(line)
    if not any1:
        L.append("- *(chưa đủ dữ liệu — cần chạy cả `pytorch` và `tensorrt`)*")

    # --- Cau 2 ---
    L.append(f"\n### 2. Với nhịp dây chuyền {cycle_ms:.0f} ms/sản phẩm, cấu hình nào đáp ứng?\n")
    have = [r for r in rows if "p95" in r]
    if have:
        L.append("Dùng **p95** chứ không phải p50: dây chuyền hỏng vì trường hợp chậm nhất, "
                 "không phải vì trường hợp trung bình.\n")
        L.append("| Cấu hình | p95 (ms) | Đáp ứng? | Nhịp nhanh nhất chịu được |")
        L.append("|---|---:|:---:|---:|")
        for r in sorted(have, key=lambda r: r["p95"]):
            ok = r["p95"] <= cycle_ms
            rate = 1000.0 / r["p95"] if r["p95"] else 0
            L.append(f"| {r['model']} / {r['backend_label']} | {r['p95']:.2f} | "
                     f"{'✅' if ok else '❌'} | {rate:.1f} sp/giây ({r['p95']:.0f} ms) |")
        passing = [r for r in have if r["p95"] <= cycle_ms]
        if passing:
            # Chon theo mAP cao nhat, NHUNG neu nhieu cau hinh co mAP gan bang nhau
            # thi lay cai nhanh nhat. Chon thuan theo mAP se de xuat PyTorch chi vi
            # hon TensorRT 0.003 mAP trong khi cham gap 2.5 lan — mot khuyen nghi to.
            top_map = max(r.get("map50", 0) for r in passing)
            TIE = 0.005
            tied = [r for r in passing if r.get("map50", 0) >= top_map - TIE]
            best = min(tied, key=lambda r: r["p95"])
            L.append(f"\n**Khuyến nghị: `{best['model']} / {best['backend_label']}`** — "
                     f"mAP@0.5 {fmt(best.get('map50'), 4)}, p95 {best['p95']:.2f} ms, "
                     f"còn dư **{cycle_ms - best['p95']:.0f} ms** biên an toàn cho chụp ảnh, "
                     f"truyền dữ liệu và dao động tải.")
            if len(tied) > 1:
                others = ", ".join(f"`{r['model']} / {r['backend_label']}`"
                                   for r in tied if r is not best)
                L.append(f"  Các cấu hình {others} có mAP chênh dưới {TIE} so với cấu hình "
                         f"tốt nhất nhưng chậm hơn, nên không được chọn.")
        else:
            L.append(f"\n**Không cấu hình nào đáp ứng {cycle_ms:.0f} ms.** Cần giảm `imgsz`, "
                     "dùng model nhỏ hơn, hoặc tăng chế độ nguồn Jetson.")
    else:
        L.append("*(chưa có dữ liệu latency)*")

    # --- Cau 3 ---
    L.append("\n### 3. YOLOv8s có đáng đổi độ trễ lấy mAP không?\n")
    n, s = by.get(("yolov8n", "tensorrt")), by.get(("yolov8s", "tensorrt"))
    if n and s and "p50" in n and "p50" in s and "map50" in n and "map50" in s:
        dl, dm = s["p50"] - n["p50"], s["map50"] - n["map50"]
        L.append(f"- Trên TensorRT FP16: YOLOv8s chậm hơn **{dl:+.2f} ms** "
                 f"({dl / n['p50'] * 100:+.0f}%) và cho mAP@0.5 **{dm:+.4f}**.")
        if dm > 0:
            L.append(f"- Đổi lại: mỗi **+0.001 mAP** tốn thêm **{dl / (dm * 1000):.2f} ms**.")
        if s.get("p95", 1e9) <= cycle_ms and dm > 0.01:
            L.append("- **Đáng.** YOLOv8s vẫn nằm trong nhịp dây chuyền và mAP cao hơn rõ rệt.")
        elif dm <= 0.01:
            L.append("- **Không đáng.** Chênh lệch mAP quá nhỏ so với chi phí độ trễ.")
        else:
            L.append("- **Không đáng** nếu phải giữ nhịp hiện tại — YOLOv8s vượt ngân sách thời gian.")
    else:
        L.append("*(cần cả yolov8n và yolov8s ở định dạng TensorRT)*")
    return L


def build_report(payload: dict, rows: list[dict], charts: list[str], cycle_ms: float) -> str:
    dev = payload.get("device", {})
    pm = dev.get("power_mode", {})
    libs = dev.get("libraries", {})
    L: list[str] = []

    L.append("# Báo cáo benchmark — ba định dạng runtime trên Jetson Orin Nano\n")
    # Dung thoi diem DO (nam trong benchmark.json) chu khong phai thoi diem sinh
    # bao cao: sinh lai bao cao tu cung mot file JSON phai cho ra file giong het,
    # neu khong thi moi lan chay 'make report' lai tao mot diff gia.
    measured = payload.get("benchmarked_utc") or dev.get("collected_utc") or "?"
    L.append(f"*Sinh tự động bởi `ivid.benchmark.report` từ `results/benchmark.json`. "
             f"Số liệu đo lúc {measured}. Mọi con số đều truy được về file JSON đó.*\n")

    L.append("## Điều kiện đo (FR-14)\n")
    L.append("| Hạng mục | Giá trị |")
    L.append("|---|---|")
    L.append(f"| Thiết bị | {dev.get('board_model') or dev.get('hostname', '?')} |")
    L.append(f"| L4T / JetPack | {(dev.get('l4t_release') or '?').strip('# ')} / {dev.get('jetpack') or '?'} |")
    L.append(f"| CUDA | {dev.get('cuda') or '?'} |")
    L.append(f"| TensorRT | {libs.get('tensorrt', '?')} |")
    L.append(f"| PyTorch | {libs.get('torch', '?')} |")
    L.append(f"| ONNX Runtime | {libs.get('onnxruntime', '?')} |")
    L.append(f"| Chế độ nguồn | **{pm.get('current_name', '?')}** (id={pm.get('current_id')}), "
             f"các chế độ có sẵn: {', '.join(pm.get('available', {}).values()) or '?'} |")
    L.append(f"| Nhiệt độ trung bình trước khi đo | {dev.get('temperature_mean_c', '?')} °C |")
    cfg = payload.get("config", {})
    L.append(f"| Warm-up / số phiên | {cfg.get('warmup', '?')} lần / {cfg.get('sessions', '?')} phiên |")
    im = payload.get("images", {})
    L.append(f"| Tập đo | {im.get('count', '?')} ảnh ({im.get('split', '?')}), cùng thứ tự cho mọi định dạng |")
    L.append("")
    L.append("> Chế độ nguồn được cố định trong suốt phiên đo (ràng buộc C3). Đổi chế độ "
             "nguồn là đổi kết quả — mọi so sánh trong báo cáo này chỉ có giá trị ở chế độ ghi trên.\n")

    L.append("## Bảng kết quả chính\n")
    L.extend(main_table(rows))
    L.append("")

    warned = [r for r in rows if r.get("warn")]
    if warned:
        L.append("> ⚠️ **Cảnh báo về tính so sánh được:**")
        for r in warned:
            L.append(f"> - `{r['model']} / {r['backend_label']}`: {r['warn']}")
        L.append("")

    L.append("## Thời gian đi đâu (FR-11)\n")
    L.append("| Model | Runtime | Tiền xử lý | Inference | Hậu xử lý (NMS) | Tổng | "
             "Inference chiếm |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        if "p50" not in r:
            continue
        share = r["infer"] / r["p50"] * 100 if r["p50"] else 0
        L.append(f"| {r['model']} | {r['backend_label']} | {r['pre']:.2f} | {r['infer']:.2f} | "
                 f"{r['post']:.2f} | {r['p50']:.2f} | {share:.0f}% |")
    L.append("")
    L.append("> Tiền xử lý và hậu xử lý dùng **cùng một hàm NumPy** cho cả ba định dạng "
             "(`ivid.preprocess`, `ivid.postprocess`). Nhờ vậy chênh lệch giữa các dòng "
             "chỉ đến từ bước inference — đúng mục đích của bảng này.\n")

    L.append("## Tài nguyên (FR-12)\n")
    L.append("| Model | Runtime | Model size | RSS đỉnh | RAM hệ thống tăng thêm | "
             "Bộ cấp phát của torch |")
    L.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        if "p50" not in r:
            continue
        L.append(f"| {r['model']} | {r['backend_label']} | {fmt(r.get('size_mb'), 1)} MB | "
                 f"{fmt(r.get('rss_mb'), 0)} MB | {fmt(r.get('sys_delta_mb'), 0)} MB | "
                 f"{fmt(r.get('gpu_mb'), 0)} MB |")
    L.append("")
    L.append("> **Đọc hai cột cuối thế nào.** Jetson dùng *bộ nhớ hợp nhất*: CPU và GPU chia "
             "nhau cùng 8 GB DRAM, không có VRAM rời. Vì vậy **RAM hệ thống tăng thêm** mới là "
             "con số phản ánh chi phí bộ nhớ thật của mỗi runtime.")
    L.append(">")
    L.append("> Cột cuối chỉ đếm phần do **chính PyTorch** cấp phát. Với ONNX Runtime nó gần "
             "bằng 0 vì ORT tự quản lý bộ nhớ GPU; với TensorRT nó chỉ đếm buffer vào/ra chứ "
             "không đếm bộ nhớ của engine. Cột này để chẩn đoán, **không dùng để so sánh "
             "giữa các runtime**.\n")

    L.append("## Tính lặp lại (NFR-02)\n")
    L.append("| Model | Runtime | Độ lệch p50 giữa các phiên | Đạt ≤ 10%? |")
    L.append("|---|---|---:|:---:|")
    for r in rows:
        if r.get("spread") is None:
            continue
        L.append(f"| {r['model']} | {r['backend_label']} | {r['spread']:.2f}% | "
                 f"{'✅' if r['spread'] <= 10 else '❌'} |")
    L.append("")

    if charts:
        L.append("## Biểu đồ\n")
        for c in charts:
            L.append(f"![{c}](images/{c})\n")

    L.append("## Kết luận (SRS §7.3)\n")
    L.extend(conclusions(rows, cycle_ms))

    per_class = next((r for r in rows if r.get("per_class")), None)
    if per_class:
        L.append("\n## mAP theo lớp\n")
        classes = sorted(per_class["per_class"])
        L.append("| Cấu hình | " + " | ".join(f"`{c}`" for c in classes) + " |")
        L.append("|---|" + "---:|" * len(classes))
        for r in rows:
            if not r.get("per_class"):
                continue
            cells = [fmt(r["per_class"].get(c, {}).get("mAP50"), 3) for c in classes]
            L.append(f"| {r['model']} / {r['backend_label']} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("> Lớp nào thấp hẳn thì xem `docs/dataset.md`: `pitted_surface` có khung "
                 "phủ hơn nửa ảnh và `crazing` là dạng vết nứt lan toả không có biên rõ — "
                 "mAP thấp ở các lớp này là đặc tính dataset, không phải lỗi pipeline.\n")

    missing = [r for r in rows if "p50" not in r or "map50" not in r]
    if missing:
        L.append("## Phần còn thiếu\n")
        for r in missing:
            what = []
            if "p50" not in r:
                what.append(f"latency ({r.get('latency_missing')})")
            if "map50" not in r:
                what.append(f"mAP ({r.get('accuracy_missing')})")
            L.append(f"- `{r['model']} / {r['backend_label']}`: " + "; ".join(what))
        L.append("")

    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/benchmark.json")
    ap.add_argument("--out", default="docs/benchmark-report.md")
    ap.add_argument("--images", default="docs/images")
    ap.add_argument("--cycle-ms", type=float, default=200.0,
                    help="nhip day chuyen de danh gia (SRS §7.3 cau 2)")
    ap.add_argument("--update-readme", action="store_true",
                    help="chen bang ket qua vao README giua hai moc IVID_TABLE")
    a = ap.parse_args()

    root = repo_root()
    src = root / a.input
    if not src.exists():
        print(f"[loi] khong thay {src} — chay ivid.benchmark.latency truoc", file=sys.stderr)
        return 1
    payload = json.loads(src.read_text(encoding="utf-8"))
    rows = rows_from(payload)
    if not rows:
        print("[loi] file benchmark khong co ket qua nao", file=sys.stderr)
        return 1

    charts = make_charts(rows, root / a.images)
    md = build_report(payload, rows, charts, a.cycle_ms)
    out = root / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print("\n".join(main_table(rows)))
    print(f"\n[report] bao cao -> {a.out}")
    for c in charts:
        print(f"[report] bieu do -> {a.images}/{c}")

    if a.update_readme:
        readme = root / "README.md"
        text = readme.read_text(encoding="utf-8")
        block = ("<!-- IVID_TABLE_START -->\n" + "\n".join(main_table(rows))
                 + "\n\n![latency](docs/images/latency_comparison.png)\n"
                 + "<!-- IVID_TABLE_END -->")
        if "<!-- IVID_TABLE_START -->" in text:
            head, rest = text.split("<!-- IVID_TABLE_START -->", 1)
            _, tail = rest.split("<!-- IVID_TABLE_END -->", 1)
            readme.write_text(head + block + tail, encoding="utf-8")
            print("[report] da cap nhat bang trong README.md")
        else:
            print("[report] README chua co moc <!-- IVID_TABLE_START --> — bo qua")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
