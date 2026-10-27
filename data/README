# Data Provenance & Preprocessing: Breast Cancer Wisconsin (Diagnostic)

## 1. Nguồn gốc dữ liệu (Provenance & Citation)
- **Tên bộ dữ liệu:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Nguồn công bố:** UCI Machine Learning Repository / National Institutes of Health (NIH)
- **Tác giả:** Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (University of Wisconsin-Madison, 1995)
- **Trích dẫn khoa học (Citation):**
  > Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology*, volume 1905, pages 861-870.
- **Giấy phép (License):** Creative Commons Attribution 4.0 International (CC BY 4.0) - Cho phép sử dụng và chia sẻ công khai hoàn toàn hợp pháp cho mục đích nghiên cứu và học thuật.
- **Điểm cộng dữ liệu:** Đáp ứng tiêu chí **Dataset công bố/benchmark (+0.25 điểm)** theo quy định của đề bài.

## 2. Đặc tả dữ liệu (Schema & Characteristics)
- **Số lượng mẫu:** 569 mẫu (357 Lành tính - Benign, 212 Ác tính - Malignant).
- **Số lượng đặc trưng:** 30 đặc trưng số thực liên tục (bán kính tế bào, kết cấu, chu vi, diện tích, độ mịn, độ nén, độ lõm, điểm lõm, tính đối xứng, kích thước phân dạng).
- **Tỷ lệ mất cân bằng nhãn:** ~62.7% Lành tính vs 37.3% Ác tính.
- **Giá trị khuyết thiếu (Missing values):** 0 (Dữ liệu hoàn chỉnh).

## 3. Quy trình tiền xử lý & Chống rò rỉ dữ liệu (Anti-Leakage Protocol)
1. **Phân chia dữ liệu (Data Splitting):**
   - Tập huấn luyện (Train set): 70% (398 mẫu).
   - Tập kiểm định (Validation set): 15% (85 mẫu) - Dùng để đánh giá hàm mục tiêu $f(x)$ trong quá trình tối ưu.
   - Tập kiểm thử độc lập (Test set): 15% (86 mẫu) - Được niêm phong hoàn toàn, chỉ dùng duy nhất 1 lần để đánh giá nghiệm cuối cùng $x^*$.
   - Sử dụng `Stratified Sampling` với `random_state` cố định để giữ nguyên phân phối nhãn qua các tập.
2. **Chuẩn hóa đặc trưng:**
   - Sử dụng `StandardScaler`.
   - **Quy tắc nghiêm ngặt:** `scaler.fit()` chỉ thực hiện trên tập Train, sau đó dùng để `transform()` lên Validation và Test set nhằm ngăn chặn hoàn toàn rò rỉ thông tin (data leakage).
