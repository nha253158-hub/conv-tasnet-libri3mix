# Conv-TasNet — Tách 3 giọng nói chồng lấn (Speech Separation)

Tách một bản ghi âm có **3 người nói cùng lúc** thành 3 file giọng riêng biệt — bài toán *"cocktail party"* kinh điển của xử lý tiếng nói. Mình **huấn luyện Conv-TasNet from scratch** (khởi tạo ngẫu nhiên, không dùng trọng số pretrained) dựa trên bản cài đặt kiến trúc của [Asteroid](https://github.com/asteroid-team/asteroid); toàn bộ pipeline huấn luyện và đánh giá là mình tự xây. Mô hình đạt **SI-SNRi 12,30 dB** trên 3000 mẫu test độc lập.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c) ![Asteroid](https://img.shields.io/badge/Asteroid-toolkit-7B2FF7) ![License](https://img.shields.io/badge/License-MIT-green)

**Tech stack:** Python · PyTorch · torchaudio · Asteroid · NumPy/Pandas · Matplotlib
**Lĩnh vực:** Deep Learning · Xử lý tín hiệu tiếng nói · Tách nguồn mù (Blind Source Separation)

![So sánh trước và sau khi tách](assets/comparison_bar.png)
*Từ một bản trộn ≈ −3,4 dB, model tách ra 3 giọng đạt ≈ +8,9 dB SI-SNR (cải thiện hơn 12 dB).*

## Điểm nổi bật

- **Xây và train từ đầu** một mạng Conv-TasNet để tách **3 nguồn** — khó hơn đáng kể so với bài tách 2 nguồn thường gặp, vì không gian tìm kiếm lớn hơn và phổ tần chồng nhau nhiều hơn.
- **Pipeline trọn vẹn:** chuẩn bị dữ liệu → huấn luyện 200 epoch (có cơ chế train nối tiếp) → đánh giá định lượng trên 3000 mẫu bằng **6 chỉ số** (SI-SNR, SI-SNRi, SDR, SDRi, PESQ, STOI).
- **Kết quả:** SI-SNRi **12,30 dB**, SDRi **12,77 dB**, STOI **0,85**. Khoảng **83%** số mẫu được cải thiện trên 10 dB.
- **Reproduce được:** notebook đánh giá load checkpoint (host trên Kaggle Dataset) và chạy thẳng ra số liệu + biểu đồ, không cần train lại.
- **Có demo nghe trực tiếp:** đưa mixture vào, lấy ra 3 giọng đã tách (xem `assets/audio_samples/`).

## Kết quả

Đánh giá trên **Libri3Mix `sep_clean`, `wav16k/min/test`** — 3000 file, mỗi file 3 người nói. Các chỉ số tính theo hoán vị bất biến (PIT) qua `asteroid.metrics.get_metrics`.

| Chỉ số           | Mean      | Std  |
| ---------------- | --------- | ---- |
| SI-SNR (dB)      | 8,93      | 2,71 |
| **SI-SNRi (dB)** | **12,30** | 2,73 |
| SDR (dB)         | 9,55      | 2,62 |
| **SDRi (dB)**    | **12,77** | 2,65 |
| PESQ             | 1,54      | 0,17 |
| STOI             | 0,85      | 0,05 |

SI-SNRi (mức cải thiện so với đầu vào) là chỉ số chính. Xét phân bố: ~73% số mẫu đạt 10–15 dB, hơn 10% trên 15 dB, chỉ ~2,6% dưới 5 dB (rơi vào các đoạn mà các giọng quá giống nhau về cao độ).

Một mốc để tham chiếu: Conv-TasNet bản gốc tách **2** người (WSJ0-2mix) đạt ~15 dB SI-SNRi. Bài toán **3** người ở đây khó hơn nhiều, nên 12,30 dB là kết quả tốt cho kiến trúc và tác vụ này.

![Dạng sóng trước và sau khi tách](assets/waveform_visualization.png)
*Hàng trên: bản trộn đầu vào. Giữa: 3 giọng gốc. Dưới: 3 giọng model tách ra.*

## Mình đã làm gì

Đây là kiến trúc có sẵn (Conv-TasNet), nhưng toàn bộ phần dựng, huấn luyện và đánh giá là mình tự làm:

- **Thiết kế cấu hình model** Conv-TasNet (encoder 1-D / khối TCN / decoder) cho 3 nguồn ở 16 kHz.
- **Viết vòng lặp huấn luyện từ đầu:** loss SI-SDR kết hợp Permutation-Invariant Training, Adam, `ReduceLROnPlateau`, gradient clipping, early stopping.
- **Tự viết pipeline đánh giá độc lập:** dataset loader cho tập test Libri3Mix, tính 6 chỉ số có PIT, xuất CSV + biểu đồ phân phối + audio mẫu.
- **Phân tích kết quả:** thống kê phân bố theo từng mức chất lượng và theo độ dài tín hiệu để hiểu model mạnh/yếu ở đâu.

## Vài quyết định kỹ thuật đáng chú ý

- **Permutation-Invariant Training (PIT).** Model xuất 3 nguồn theo thứ tự bất kỳ, không cố định nguồn nào là s1/s2/s3. Cả loss lúc train lẫn lúc đo đều phải ghép cặp theo hoán vị tối ưu — bỏ qua bước này là số liệu sai hoàn toàn.
- **Huấn luyện nối tiếp qua nhiều phiên.** Train trên Kaggle mà mỗi phiên giới hạn 12 tiếng, nên mình lưu checkpoint mỗi epoch gồm cả optimizer + scheduler + lịch sử loss. Hết giờ thì nạp lại đúng trạng thái và train tiếp, không mất tiến độ.
- **Xử lý state dict của DataParallel.** Train đa GPU khiến key bị thêm tiền tố `module.`; mình strip prefix khi load để checkpoint tái sử dụng được ở mọi nơi (1 GPU hoặc CPU).
- **Tách biệt train và eval.** Phần đánh giá là notebook riêng, chỉ load checkpoint rồi chạy độc lập trên tập test — đảm bảo số liệu trong báo cáo là khách quan và reproduce được.

## Kiến trúc & cấu hình

Conv-TasNet làm việc thẳng trên miền thời gian: encoder 1-D học cách biểu diễn tín hiệu, khối TCN ước lượng mặt nạ cho từng nguồn, decoder dựng lại từng giọng. Dùng bản cài đặt trong [Asteroid](https://github.com/asteroid-team/asteroid).

| Tham số                              | Giá trị         |     | Huấn luyện | Giá trị           |
| ------------------------------------ | --------------- | --- | ---------- | ----------------- |
| `n_src`                              | 3               |     | Optimizer  | Adam (lr 1e-3)    |
| Sample rate                          | 16 kHz          |     | Loss       | SI-SDR + PIT      |
| `n_filters`                          | 512             |     | Scheduler  | ReduceLROnPlateau |
| kernel / stride                      | 32 / 16         |     | Grad clip  | max-norm 5.0      |
| `bn_chan` / `hid_chan` / `skip_chan` | 128 / 512 / 128 |     | Segment    | 3 giây, batch 4   |
| `n_blocks` × `n_repeats`             | 8 × 3           |     | Epoch      | tối đa 200        |

Val loss thấp nhất là **−9,30 dB**, đạt quanh epoch 191; sau ~epoch 150 thì gần như đi ngang (đã hội tụ).

## Cài đặt & chạy

```bash
git clone https://github.com/nha253158-hub/conv-tasnet-libri3mix.git
cd conv-tasnet-libri3mix
pip install -r requirements.txt
```

Dữ liệu **Libri3Mix** (16 kHz, mode `min`, task `sep_clean`) sinh bằng repo gốc [LibriMix](https://github.com/JorisCos/LibriMix). Thư mục test có dạng `test/{mix_clean, s1, s2, s3}/`.

- **Train:** mở `notebooks/train_conv_tasnet.ipynb`, sửa `csv_dir`, chạy hết. Code tự lưu/khôi phục checkpoint nên train nối tiếp được.
- **Đánh giá:** mở `notebooks/evaluate_conv_tasnet.ipynb`, sửa `CHECKPOINT_PATH` + `LIBRI3MIX_TEST_DIR`, chạy hết để ra bảng số liệu và hình. Đặt `MAX_SAMPLES=100` để test nhanh.

Tách thử một file bất kỳ (16 kHz):

```python
import torch, torchaudio
from asteroid.models import ConvTasNet

model = ConvTasNet(n_src=3, sample_rate=16000, kernel_size=32, n_filters=512,
                   stride=16, bn_chan=128, hid_chan=512, skip_chan=128,
                   n_blocks=8, n_repeats=3, mask_act="relu")

ckpt = torch.load("last_checkpoint.pth", map_location="cpu")
state = ckpt.get("model_state_dict", ckpt)
state = {k.replace("module.", "", 1): v for k, v in state.items()}  # bỏ prefix DataParallel
model.load_state_dict(state); model.eval()

mix, sr = torchaudio.load("mixture.wav")
with torch.no_grad():
    est = model(mix.unsqueeze(0)).squeeze(0)   # (3, T)
for i in range(3):
    torchaudio.save(f"estimated_s{i+1}.wav", est[i:i+1], 16000)
```

## Hạn chế & hướng phát triển

- Ở các đoạn mà các giọng quá giống nhau (cùng giới, cao độ gần) vẫn còn xuyên âm — PESQ ~1,5 phản ánh điều này.
- Model chạy offline, chưa real-time. Có thể thử biến thể causal cho ứng dụng thời gian thực.
- Hướng tiếp theo: thử các kiến trúc mới hơn (DPRNN, SepFormer), bổ sung dữ liệu có nhiễu/vọng thực tế, và tăng quy mô model.

## Notebook & demo

|          | Mô tả                                             | Notebook                                                           |
| -------- | ------------------------------------------------- | ----------------------------------------------------------------- |
| Train    | Conv-TasNet trên Libri3Mix sep_clean (200 epoch)  | [`train_conv_tasnet.ipynb`](notebooks/train_conv_tasnet.ipynb)       |
| Đánh giá | 6 chỉ số + biểu đồ trên 3000 file                 | [`evaluate_conv_tasnet.ipynb`](notebooks/evaluate_conv_tasnet.ipynb) |

GitHub render trực tiếp `.ipynb` nên có thể xem code và kết quả ngay trên trình duyệt. Checkpoint được host thành Kaggle Dataset và gắn sẵn vào notebook đánh giá, nên chạy lại được end-to-end mà không cần train lại.

## Tham khảo

```bibtex
@article{luo2019conv,
  title   = {Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation},
  author  = {Luo, Yi and Mesgarani, Nima},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2019}
}
```

Cảm ơn [Asteroid](https://github.com/asteroid-team/asteroid) và [LibriMix](https://github.com/JorisCos/LibriMix).

## License

[MIT](LICENSE)
