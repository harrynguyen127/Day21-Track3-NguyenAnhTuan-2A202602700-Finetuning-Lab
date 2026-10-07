# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Anh Tuấn  **MSSV**: 2A202602700  **Ngày**: 2026-10-07  
**Tier**: T4  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: T4 tier; model GPU cụ thể chưa có trong output đã chia sẻ.

> Các số liệu dưới đây được lấy từ artefact của lần chạy. Dự đoán baseline (b) theo từng ticket không được lưu trong artefact hiện có, vì vậy không suy diễn kết quả thắng/thua từng mẫu so với baseline đó.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (giá trị cấu hình T4; thống kê p95 gợi ý 256) |
| p95 token | 98 *(n=250; mean 93.1; max 101)* |
| `MASK_MODE` | `assistant-only` (theo `results/runs.csv`, run `correct`) |
| Epochs / max_steps | 30 optimizer steps; số epoch không được ghi trong artefact đã tải |
| Precision / peak VRAM | fp16 / 8.78 GB cho run `correct` |

**Template có giữ khối `<think>` không?** Có. `open_tag_present=true`, `body_present=true`; verdict của kiểm tra là `reasoning preserved — safe to train on traces`.

## 2. Mask proof (NB1)

Gatekeeper xác nhận các assert của mask proof đã qua: câu trả lời được tính loss và câu hỏi bị mask.

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39 / 94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn decoded được tính loss (trích từ `supervised_preview`):

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Mask proof cho thấy câu hỏi ở phần masked và câu trả lời ở phần supervised.

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3184.8 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 995.4 |
| (c) LoRA fine-tune | 0.970 | 0.6111 | 1.000 | 1342.6 |

Baseline (b) tốt hơn (a) trên target: 0.765 so với 0.000. Gatekeeper xác nhận prompt baseline (b) không bị sửa và (b) vượt (a). Fine-tune tăng target thêm 0.205 so với (b), nhưng regression giảm 0.180. Latency fine-tune là 1342.6 ms, cao hơn baseline (b) 995.4 ms trong phép đo này.

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable params | LR | train loss | target (NB5) | train seconds | peak VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6260 | 0.970 | 405.2 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 0.0001 | 0.5369 | 0.970 | 270.7 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.000 | 407.2 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 469.3 | 3.86 |

Gatekeeper xác nhận bốn run đều dùng 30 bước và `attn_only` có số tham số trainable gần bằng `correct` (32,456,704 so với 32,464,896). Target score lần lượt là 0.970, 0.970, 0.000 và 0.940. `autopsy.json` cũng ghi format lần lượt là 1.0, 1.0, 0.0 và 1.0; latency là 1342.6, 872.9, 5118.8 và 1738.2 ms.

### 4.1 — Vị trí adapter và rank

`attn_only` có số tham số trainable gần như bằng `correct`, nhưng rank của nó là 283 thay vì 16 để khớp ngân sách tham số. Train loss cuối của `attn_only` là 0.5369, thấp hơn 0.6260 của `correct`, trong khi target score của cả hai cùng là 0.970. Vì vậy hai cấu hình hòa nhau theo target ở tập 50 mẫu này, dù train loss khác nhau. Kết quả này không cho thấy rank cao hơn giúp target tốt hơn; nó cũng không chứng minh hai vị trí adapter luôn tương đương ngoài phép đo này.

### 4.2 — Learning rate

Hai run dùng cùng vị trí adapter, rank, số tham số và ngân sách 30 bước; learning rate của `wrong_lr` là 0.00001, thấp hơn mười lần so với 0.0001 của `correct`. Loss cuối là 1.5702 cho `wrong_lr` và 0.6260 cho `correct`; target tương ứng là 0.000 và 0.970. Đây là điểm cuối, không phải toàn bộ đường cong loss, nên artefact hiện có không cho biết diễn biến theo từng bước. Trong lần chạy này, chỉ nhìn train loss vẫn không phải thang đo cần dùng để xếp hạng chất lượng tác vụ; target score cho thấy `wrong_lr` không giải quyết được tác vụ.

### 4.3 — QLoRA

Peak VRAM của QLoRA là 3.86 GB, thấp hơn 4.92 GB so với 8.78 GB của `correct` (giảm khoảng 56% theo peak VRAM đã ghi). Train loss cuối của QLoRA là 0.7058 so với 0.6260; target là 0.940 so với 0.970, format đều là 1.0, và latency QLoRA cao hơn (1738.2 so với 1342.6 ms). Trên T4 của lần chạy này, `correct` hoàn tất ở peak 8.78 GB, nên QLoRA không cần thiết chỉ để vừa bộ nhớ trong phép đo này; nó đi kèm target thấp hơn 0.030 và latency cao hơn. Điều này ủng hộ thận trọng với QLoRA trong cấu hình cụ thể này, nhưng một run không đủ để khái quát cho model, dữ liệu hay phần cứng khác.

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.180` · `valid_trace_rate = 0.0`

Fine-tune đạt target 0.970, cao hơn baseline (b) 0.765 một mức 0.205. Tuy nhiên, regression giảm từ 0.7911 ở baseline (b) xuống 0.6111, tức giảm 0.180; Gatekeeper ghi rõ mức giảm này vượt ngưỡng cho phép 0.020. Vì vậy kết quả không qua cổng hồi quy dù chỉ số target tăng. `valid_trace_rate` là 0.0 trong kết quả được báo cáo. Kết luận phù hợp với phép đo này là fine-tune cải thiện tác vụ triage mục tiêu nhưng đồng thời làm suy giảm nhóm năng lực regression; không nên gọi run này là đạt yêu cầu triển khai nếu regression là tiêu chí bắt buộc. Không thay đổi ngưỡng hoặc tập đánh giá sau khi đã xem kết quả. Cần điều tra các ví dụ regression bị mất điểm trước khi thử huấn luyện lại; một thử nghiệm replay data có thể được cân nhắc, nhưng chưa được thực hiện trong kết quả này.

## 6. Định tính — đối chiếu fine-tune với nhãn

`qualitative.json` lưu ticket, `ft_score` và preview dự đoán fine-tune; `eval_target.jsonl` cung cấp nhãn đúng. Artefact hiện có không lưu dự đoán baseline (b) theo từng ticket, nên bảng dưới đây đối chiếu fine-tune với nhãn chứ không tuyên bố fine-tune thắng baseline (b) trên từng mẫu. Hai mẫu đạt 4/4 trường và ba mẫu đạt 3/4 trường. Preview dự đoán trong JSON bị cắt ở 90 ký tự; phần không có trong file được ghi rõ thay vì tự khôi phục.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Chuột không dây; yêu cầu trả lại; gấp; khen shop | doi_tra / cao / chuột không dây / tich_cuc | Không có dự đoán từng mẫu | 4/4 trường đúng (`ft_score=1.0`) | Khớp toàn bộ nhãn |
| 2 | Ốp lưng điện thoại; hoàn tiền; sớm; bực mình | hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc | Không có dự đoán từng mẫu | 4/4 trường đúng (`ft_score=1.0`) | Khớp toàn bộ nhãn |
| 3 | Bình giữ nhiệt; chưa thấy tiền; khi nào tiện | hoan_tien / thap / bình giữ nhiệt / tich_cuc | Không có dự đoán từng mẫu | intent=hoan_tien; urgency=trung_binh; product=bình giữ nhiệt; sentiment preview bị cắt (`ft_score=0.75`) | Sai urgency; 3/4 trường khớp |
| 4 | Nồi chiên không dầu; thiếu phụ kiện; khi nào tiện | san_pham_loi / thap / nồi chiên không dầu / trung_tinh | Không có dự đoán từng mẫu | intent=san_pham_loi; urgency=trung_binh; product=nồi chiên không dầu; sentiment preview bị cắt (`ft_score=0.75`) | Sai urgency; 3/4 trường khớp |
| 5 | Áo khoác gió; bị lỗi; khi nào tiện | san_pham_loi / thap / áo khoác gió / tich_cuc | Không có dự đoán từng mẫu | intent=san_pham_loi; urgency=trung_binh; product=áo khoác gió; sentiment preview bị cắt (`ft_score=0.75`) | Sai urgency; 3/4 trường khớp |

Ở ba trường hợp đạt 0.75 được chọn, urgency dự đoán là `trung_binh` trong khi nhãn là `thap`; điểm số cho biết ba trong bốn trường khớp. Hai ví dụ đạt 1.0 khớp đủ bốn trường. Chưa thể nhận xét mẫu nào baseline (b) xử lý tốt hơn vì không có dự đoán baseline theo từng ticket.

## 7. Kết luận & điều tôi học được

Trong lần chạy này, LoRA cải thiện điểm target từ 0.765 của base model với optimized prompt lên 0.970, nhưng không vượt qua cổng hồi quy: điểm regression giảm 0.180, trong khi ngưỡng cho phép là 0.020. Vì thế, chỉ nhìn vào target sẽ khiến kết luận quá lạc quan. Tôi chưa đề xuất deploy run này nếu yêu cầu sản phẩm là giữ năng lực regression; cần điều tra nguyên nhân giảm điểm và đánh giá biện pháp giảm hồi quy trước khi quyết định. Ở NB4, `attn_only` đạt cùng target 0.970 với `correct`, `wrong_lr` đạt 0.000, còn QLoRA đạt 0.940. QLoRA tiết kiệm peak VRAM 4.92 GB nhưng có target thấp hơn 0.030 và latency cao hơn 395.6 ms so với `correct` trong lần chạy này. Các ví dụ target được chọn cho thấy ba ca fine-tune sai trường urgency, trong khi hai ca khớp đủ bốn trường; artefact không lưu dự đoán baseline (b) từng ticket nên không thể đối chiếu thắng/thua giữa hai model theo từng mẫu. Bước tiếp theo hợp lý là thử replay data để giảm hồi quy rồi đánh giá lại trên cùng tập eval; đó là đề xuất cho thí nghiệm tiếp theo, chưa phải kết quả đã đo.

**Ba điều tôi học được từ số liệu hiện có**:
1. Điểm target tăng không đảm bảo mô hình qua cổng hồi quy: target tăng 0.205 nhưng regression giảm 0.180.
2. `attn_only` có train loss thấp hơn `correct` nhưng cùng target 0.970; train loss không thay thế được điểm tác vụ.
3. QLoRA giảm peak VRAM 4.92 GB, đồng thời target thấp hơn 0.030 và latency cao hơn trong run này.

**Nếu có thêm 2 giờ nữa, tôi sẽ:** xem các ví dụ regression giảm điểm và thử thêm replay data để đánh giá liệu regression có được giữ lại mà không làm mất mức tăng target hay không.

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
