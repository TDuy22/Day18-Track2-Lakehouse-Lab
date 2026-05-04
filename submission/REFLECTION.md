# REFLECTION

**Anti-pattern dễ vướng nhất: "Small File Problem" (Vấn đề tệp dữ liệu rác/nhỏ)**

Trong quá trình xây dựng Lakehouse, đặc biệt là ở layer Bronze khi ingest dữ liệu liên tục từ các luồng streaming hoặc micro-batch, team rất dễ vướng vào tình trạng tạo ra hàng ngàn đến hàng triệu file nhỏ (small files) trên hệ thống lưu trữ (MinIO/S3/Local). 

**Vì sao team dễ vướng phải?**
Thường team sẽ ưu tiên việc đẩy dữ liệu vào Data Lake nhanh nhất có thể để đảm bảo tính real-time mà quên đi việc bảo trì định kỳ. Quá trình append liên tục vào Delta Table mà không được nén hay gộp lại sẽ dẫn đến sự phình to của siêu dữ liệu (metadata) cũng như số lượng file vật lý.

**Hậu quả:**
- Hiệu năng truy vấn giảm sút nghiêm trọng vì engine (như Spark hay DuckDB) phải tốn quá nhiều overhead để mở và đóng file thay vì thực sự quét dữ liệu.
- Tốn kém chi phí thực thi các lệnh liệt kê (list/GET) trên Object Storage.

**Giải pháp:**
Cần thiết lập pipeline định kỳ chạy lệnh `OPTIMIZE` để gộp các file nhỏ (compaction) và `Z-ORDER` theo các trường dữ liệu hay được filter để tăng tốc độ truy vấn, tương tự như những gì đã thực hành trong notebook `02_optimize_zorder`.
