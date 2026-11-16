# THÔNG TIN BỘ DỮ LIỆU VÀ QUY TRÌNH XỬ LÝ

## 1. Nguồn dữ liệu

- **Tên:** Breast Cancer Wisconsin (Diagnostic).
- **Nguồn:** UCI Machine Learning Repository, có thể tải qua thư viện `scikit-learn`.
- **Tác giả:** W. H. Wolberg, W. N. Street, O. L. Mangasarian.
- **Trích dẫn:** Wolberg và cộng sự (1995), https://doi.org/10.24432/C5DW2B.
- **Mục đích:** minh họa phương pháp phân loại và tối ưu hóa siêu tham số, không dùng cho chẩn đoán lâm sàng.

## 2. Đặc điểm dữ liệu

- Có 569 mẫu, 30 đặc trưng dạng số, hai lớp lành tính và ác tính.
- Dữ liệu được lấy bằng `sklearn.datasets.load_breast_cancer()` trong mã nguồn, không cần tải tệp CSV riêng.

## 3. Chia dữ liệu để tránh rò rỉ thông tin

Chương trình sử dụng `train_test_split` hai lần với `stratify` và seed 42:

| Tập dữ liệu | Số mẫu | Chức năng |
|---|---:|---|
| Huấn luyện (train) | 397 | Huấn luyện Random Forest |
| Kiểm định (validation) | 86 | Tính F1 và độ trễ để tối ưu siêu tham số |
| Kiểm thử (test) | 86 | Đánh giá mô hình sau khi chọn cấu hình |

Không sử dụng nhãn của tập test trong quá trình tối ưu hoặc chọn siêu tham số. Vì Random Forest không yêu cầu chuẩn hóa biến đầu vào, dự án **không** sử dụng `StandardScaler`.

## 4. Đo độ trễ và tính tái lập

Tạo một tập 1.000 hàng bằng cách lấy mẫu có hoàn lại từ validation với bộ sinh ngẫu nhiên có seed cố định. Thử nghiệm các mô hình trên cùng tập này để so sánh thời gian dự đoán. Bản ghi chi tiết ở `results/experiment_metadata.json`.

## 5. Ghi chú sử dụng

Nguồn dữ liệu được công bố công khai cho mục đích nghiên cứu. Khi trích dẫn lại, ghi rõ UCI và tác giả. Không diễn giải kết quả học thuật như bằng chứng đủ để triển khai y tế.
