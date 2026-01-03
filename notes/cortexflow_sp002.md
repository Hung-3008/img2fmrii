# CortexFlow SP002 Thiết kế: Flat Conditional Flow (Phiên bản Đơn giản hóa)

**Ngày:** 2026-01-03
**Trạng thái:** Đề xuất (Draft)
**Mục tiêu:** Đơn giản hóa kiến trúc CortexFlow gốc để giảm độ phức tạp khi hiện thực, giảm yêu cầu bộ nhớ, và kiểm chứng nhanh hiệu quả của mô hình Flow trên dữ liệu NSD.

---

## 1. Động lực (Motivation)
Đề xuất CortexFlow gốc (SP001) rất mạnh về mặt lý thuyết giải phẫu thần kinh (neuro-anatomy driven), nhưng gặp các rào cản lớn khi triển khai:
*   **Phức tạp kỹ thuật:** Việc thiết kế Flow theo cấu trúc phân cấp cững nhắc (V1 -> V2 -> IT) đòi hỏi việc map dữ liệu (ROI mapping) cực kỳ chính xác và custom layers phức tạp.
*   **Chi phí tính toán:** Duy trì spatial structure qua nhiều lớp Flow tốn kém bộ nhớ.

**Phiên bản SP002 - "CortexFlow-Flat"** sẽ loại bỏ các ràng buộc cấu trúc không gian cứng nhắc, chuyển sang cách tiếp cận "Data-Driven" hoàn toàn, tương tự như cách GLOW hoạt động trên ảnh nhưng áp dụng cho vector voxel 1D.

## 2. Kiến trúc Đề xuất (Architecture)

Mô hình sẽ là một **Conditional Normalizing Flow** (dựa trên RealNVP hoặc Glow) ánh xạ trực tiếp từ không gian nhiễu (Gaussian Noise) sang không gian Voxel, được điều kiện hóa bởi vector đặc trưng hình ảnh (CLIP Embedding).

### 2.1 Đầu vào & Đầu ra
*   **Input (Condition):** CLIP Image Embedding $c \in \mathbb{R}^{768}$ (từ ViT-L/14).
*   **Base Distribution:** Latent noise $z \sim \mathcal{N}(0, I)$, cùng chiều với số lượng voxel.
*   **Output:** Voxel vector $x \in \mathbb{R}^N$ (với $N \approx 10,000 - 20,000$ voxels thuộc Visual Cortex, đã được làm phẳng).

### 2.2 Luồng xử lý (Flow Steps)
Thay vì chia tách theo layer V1/V2, ta coi toàn bộ ROI là một vector lớn và dùng các khối Flow tiêu chuẩn:

1.  **Preprocessing:** PCA giảm chiều (Optional) hoặc giữ nguyên full voxels nếu VRAM cho phép (NSD có ~15k voxels visual, có thể train được trên GPU 24GB+).
2.  **Coupling Layers (Affine Injectors):**
    *   Sử dụng kiến trúc **RealNVP** cải tiến.
    *   Chia vector voxel $x$ thành 2 phần $x_a, x_b$ (checkerboard masking hoặc channel-wise split).
    *   Phép biến đổi:
        $$x_b' = x_b \odot \exp(s(x_a, c)) + t(x_a, c)$$
        $$x_a' = x_a$$
    *   Trong đó $s(\cdot)$ (scale) và $t(\cdot)$ (translate) là các mạng MLP đơn giản (2-3 hidden layers).
    *   **Conditioning:** Vector CLIP $c$ được nối (concat) vào input của mạng $s$ và $t$, hoặc điều khiển qua cơ chế FiLM layer.

3.  **Cross-Subject Adaptation (Đơn giản hóa):**
    *   Thay vì learnable embedding phức tạp, sử dụng một **Linear Adapter** đơn giản ở tầng đầu ra cuối cùng để căn chỉnh data của từng subject về không gian chung trước khi đưa vào Flow, hoặc training riêng biệt cho từng subject (Single-Subject Model) trong pha 1 để validate.

## 3. Các thay đổi chính so với bản gốc

| Đặc điểm | CortexFlow Gốc (Complex) | **CortexFlow SP002 (Simple)** |
| :--- | :--- | :--- |
| **Cấu trúc không gian** | Hierarchical (V1 $\to$ V2 $\to$ IT) | **Flat (Vector 1D toàn cục)** |
| **Spatial Prior** | Receptive Fields, Convolutional Flow | **Fully Connected / MLP Coupling** |
| **Mục tiêu tối ưu** | MLE + RSA Loss + Receptive Field Constraint | **Chủ yếu là MLE (Maximum Likelihood)** |
| **Input Data** | 3D Volume hoặc Cortical Surface Map | **Masked 1D Vector (chỉ ROI cần thiết)** |
| **Độ khó hiện thực** | Rất cao (Custom CUDA/Flow layers?) | **Trung bình (Dùng thư viện Flow chuẩn)** |

## 4. Kế hoạch Thử nghiệm (Validation Plan)

1.  **Bước 1 (Data Prep):** Trích xuất dữ liệu NSD, mask lấy các vùng Visual Cortex (V1-V4, ventral temporal), làm phẳng thành vector 1D. Chuẩn hóa dữ liệu (Z-score).
2.  **Bước 2 (Baseline Training):** Huấn luyện CortexFlow-SP002 trên dữ liệu của **Subject 1** (nhiều dữ liệu nhất).
3.  **Bước 3 (Evaluation):**
    *   *Reconstruction:* Đo tương quan (Pearson correlation) giữa fMRI sinh ra và ground truth trên tập *test set*.
    *   *Diversity:* Kiểm tra variance của các mẫu sinh ra từ cùng một ảnh input (khác noise $z$).
    *   *Interpretability:* Thử nội suy (interpolate) trong không gian $z$ và quan sát sự thay đổi trơn tru của tín hiệu voxel.

## 5. Kết luận
CortexFlow SP002 hy sinh tính chính xác về giải phẫu học (anatomical fidelity) của phiên bản gốc để đổi lấy sự **đơn giản và khả thi (feasibility)**. Đây là bước đệm cần thiết: nếu SP002 hoạt động tốt (tương quan cao hơn SynBrain/Regression baseline), ta mới có cơ sở để đầu tư phát triển phiên bản phân cấp phức tạp (SP001).
