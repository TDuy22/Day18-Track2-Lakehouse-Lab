# Architecture Brief — Multimodal RAG trên 10 Triệu Tài Liệu Pháp Lý Việt Nam

**Topic D | Author: Trần Lâm Duy | Lab 18 — Bonus Challenge**

---

## 1. Problem Statement

Một văn phòng luật lớn tại Việt Nam cần hệ thống RAG (Retrieval-Augmented Generation) phục vụ tra cứu án lệ, văn bản quy phạm pháp luật và hợp đồng nội bộ trên kho tài liệu **10 triệu PDF** gồm: text thuần, ảnh scan (CCCD, biên bản), và bảng dữ liệu (phụ lục hợp đồng). Tổng dung lượng tài liệu gốc ước tính **~50 TB**.

**Constraints cứng (hard constraints):**
- Search latency p95 **< 200 ms** trên kho **30 tỷ token chunks** (~1.5 tỉ chunks × 20 tokens/chunk trung bình).
- Embeddings **sẽ được regenerate ≥ 2 lần** khi nâng cấp embedding model (ví dụ: từ `text-embedding-ada-002` → `text-embedding-3-large` → mô hình nội bộ fine-tuned).
- Kết quả retrieval phải **reproducible sau 5 năm**: khi một bản án vào năm 2025 trích dẫn "chunk #xyz từ Bộ luật Dân sự bản 2015, embedding model v1", thì năm 2030 vẫn phải retrieve ra đúng chunk đó.
- **Data governance pháp lý**: tài liệu có thể bị tòa án thu hồi hoặc phán quyết bị sửa đổi — cần cơ chế hard-delete có audit trail.

**Tại sao khó:** Bài toán kết hợp 3 thách thức đồng thời: (1) scale cực lớn (30B tokens), (2) multimodal input (text/image/table), (3) reproducibility yêu cầu versioning nghiêm ngặt cho cả document lẫn embedding — trong khi hầu hết vector DB chỉ lưu "version mới nhất".

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  INGESTION PATH                                                                      │
│                                                                                      │
│  [PDF Upload API]  ──►  [Kafka / File Queue]  ──►  [Parser Workers]                 │
│       10M PDFs              (event log)              ├─ pdfplumber  (text+table)     │
│                                                      └─ Tesseract OCR (image pages)  │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────── MEDALLION LAKEHOUSE (MinIO / S3) ───────────────────────────┐
│                                                                                      │
│  ╔══════════════════════════════════════════════════════════════════════════════╗    │
│  ║  BRONZE  (Delta Lake — append-only, raw)                                    ║    │
│  ║  • raw_documents        : doc_id, s3_raw_pdf_uri, ingested_at, sha256       ║    │
│  ║  • raw_pages            : doc_id, page_no, modality (text/image/table),     ║    │
│  ║                           raw_text, ocr_text, png_s3_uri                    ║    │
│  ║  ──── partition: ingested_date  |  format: Parquet+Snappy                   ║    │
│  ╚══════════════════════════════════════════════════════════════════════════════╝    │
│                               │ Delta MERGE (dedup by sha256)                        │
│                               ▼                                                      │
│  ╔══════════════════════════════════════════════════════════════════════════════╗    │
│  ║  SILVER  (Delta Lake — cleaned, chunked, normalized)                        ║    │
│  ║  • clean_chunks         : chunk_id (UUID), doc_id, page_no, chunk_text,     ║    │
│  ║                           modality, token_count, created_at, is_deleted     ║    │
│  ║  • doc_metadata         : doc_id, title, court, issued_date, doc_type,      ║    │
│  ║                           jurisdiction, revision_no, withdrawn_at           ║    │
│  ║  ──── partition: doc_type / issued_year  |  OPTIMIZE + Z-ORDER(doc_id)      ║    │
│  ╚══════════════════════════════════════════════════════════════════════════════╝    │
│                               │ Embedding pipeline (versioned)                       │
│                               ▼                                                      │
│  ╔══════════════════════════════════════════════════════════════════════════════╗    │
│  ║  GOLD  (Lance format — vector store, versioned per embedding model)         ║    │
│  ║  • embeddings_v{N}      : chunk_id, doc_id, embedding_model_id,             ║    │
│  ║                           vector [float32 × 1536], created_at               ║    │
│  ║  • model_registry       : embedding_model_id, model_name, dimension,        ║    │
│  ║  (Delta)                  checksum, deprecated_at, successor_model_id       ║    │
│  ║  ──── IVF_PQ index (nlist=4096, M=64)  |  Lance versioned snapshots         ║    │
│  ╚══════════════════════════════════════════════════════════════════════════════╝    │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                         │                              │
         QUERY PATH      │                              │ AUDIT / LINEAGE
                         ▼                              ▼
              ┌─────────────────┐            ┌──────────────────────┐
              │  Vector Search  │            │  Delta: audit_reads   │
              │  Lance ANN      │            │  (who read which doc, │
              │  p95 < 120 ms   │            │   when, for what case)│
              └────────┬────────┘            └──────────────────────┘
                       │ top-K chunk_ids
                       ▼
              ┌─────────────────┐
              │  Rerank + LLM   │
              │  (GPT-4o / Gemini│
              │   Pro via API)  │
              └─────────────────┘
                       │
                       ▼
              [Legal Answer + Source Citations with version pinning]
```

---

## 3. Các Quyết Định Kiến Trúc Chính

### Quyết định 1 — Table Format cho metadata layer: **Delta Lake**

**Tôi chọn Delta Lake** cho Bronze và Silver (metadata, chunks, audit tables).

- **Loại Iceberg** vì: Delta Lake hỗ trợ `deletion vectors` (soft-delete không cần rewrite toàn bộ file Parquet), critical khi tòa án thu hồi tài liệu — chỉ mark deletion vector thay vì scan 50 TB. Iceberg yêu cầu copy-on-write hoặc MOR merge, đắt hơn ở quy mô này.
- **Loại plain Parquet** vì: Không có ACID transaction log — nếu embedding pipeline crash giữa chừng khi update 1.5B chunks, không thể rollback. Delta `_delta_log` cho phép `RESTORE TABLE TO VERSION AS OF N` để quay lại trạng thái trước crash trong < 30 giây.
- **Con số cụ thể:** Silver `clean_chunks` ước tính ~1.5B rows × ~2 KB/row (chunk text + metadata) ≈ **3 TB**. Với deletion vectors, việc thu hồi 10.000 chunks của một bộ luật bị sửa đổi chỉ tốn ~50 ms thay vì rewrite 3 TB.

### Quyết định 2 — Vector Store Format: **Lance (không phải Delta)**

**Tôi chọn Lance** cho Gold layer (embeddings + ANN index).

- **Loại Delta Lake + external FAISS** vì: Delta lưu vector dưới dạng Parquet binary blob — mỗi ANN query phải deserialize toàn bộ column, không có native index. FAISS index là external file, không được versioned cùng data, dẫn đến index/data drift khi re-embed.
- **Loại pgvector (PostgreSQL)** vì: Không thể scale tới 30B token chunks trên single-node PostgreSQL. Horizontal sharding pgvector cực kỳ phức tạp, và không có native time-travel để pin embedding version cụ thể.
- **Lance** cung cấp: (a) columnar format native cho vector, (b) versioned snapshots built-in (mỗi re-embed tạo version mới, version cũ vẫn queryable — reproducibility requirement met), (c) IVF_PQ index stored inside Lance dataset, (d) DuckDB integration để join Lance vector results với Delta metadata trong một pipeline.
- **Con số cụ thể:** 1.5B chunks × 1536 floats × 4 bytes = **~9.2 TB** cho embeddings_v1. Lance IVF_PQ với M=64 giảm xuống còn ~2.3 TB với precision loss < 2% recall@10.

### Quyết định 3 — Embedding Versioning Strategy: **Parallel Namespaces, không overwrite**

**Tôi chọn tạo table mới `embeddings_v{N}`** mỗi lần regenerate, giữ song song.

- **Loại overwrite `embeddings` table** vì: Vi phạm reproducibility constraint — một bản án năm 2025 đã lưu citation với embedding_model_id=v1, năm 2030 không thể tái hiện đúng retrieval nếu đã overwrite.
- **Loại update-in-place với version column** vì: Lance không optimize tốt cho mixed-version queries. Hơn nữa, trong quá trình regenerate (kéo dài 3–7 ngày cho 1.5B chunks), hệ thống cần serving version cũ và version mới song song (blue/green embedding).
- **Cơ chế:** `model_registry` Delta table track `embedding_model_id → active/deprecated`. Query API luôn default vào active model, nhưng có thể pin `?embedding_version=v1` cho reproducible citation lookup. Old versions được retain tối thiểu 6 năm (statute of limitations pháp lý Việt Nam).
- **FinOps trade-off:** Mỗi version tốn ~2.3 TB (compressed). 3 versions = ~7 TB Gold storage ≈ $161/tháng (S3 Standard). Chấp nhận được so với alternative là mất reproducibility.

### Quyết định 4 — Chunking Strategy: **Multimodal-aware, không uniform**

**Tôi chọn chunking theo semantic boundary, khác nhau cho mỗi modality:**

- Text pages: sliding window 512 tokens, overlap 64 tokens → `modality='text'`
- Bảng (tables): toàn bộ bảng là 1 chunk, serialize thành markdown → `modality='table'`  
- Image pages (scan): OCR text → chunk như text; đồng thời lưu `png_s3_uri` trong Bronze để future CLIP/multimodal embedding → `modality='image_ocr'`
- **Loại uniform 256-token chunking cho tất cả** vì: Table bị cắt giữa chừng mất ngữ cảnh hoàn toàn. Một điều khoản pháp lý bị cắt sai chỗ tạo ra hallucination nghiêm trọng — rủi ro pháp lý cho văn phòng luật.
- **Loại document-level embedding** vì: 30B tokens không thể fit context window của bất kỳ LLM nào hiện tại. Chunking là bắt buộc.

### Quyết định 5 — ANN Index: **IVF_PQ, không phải HNSW**

**Tôi chọn IVF_PQ (Inverted File + Product Quantization)** cho Gold layer.

- **Loại HNSW** vì: HNSW yêu cầu full graph loaded in memory — 30B vectors × 1536D với HNSW overhead ≈ **> 200 TB RAM**, không feasible. Ngoài ra, HNSW không support efficient disk-based access pattern.
- **IVF_PQ tham số:** nlist=4096 centroids (√1.5B ≈ 38K, chọn 4096 cho trade-off), M=64 sub-quantizers, nbits=8 → mỗi vector nén từ 6 KB xuống còn 64 bytes. Tổng index in-memory: 4096 × 64 × 8 bytes centroids + 1.5B × 64 bytes = **~100 GB** — fit trên một máy chủ 128 GB RAM.
- **Con số latency:** nprobe=128 → scan 128/4096 = 3.1% dataset mỗi query = ~46M vectors. Tốc độ SIMD inner product: ~2B vectors/giây/core × 8 cores = **~23 ms scan time** + 10 ms network overhead = p95 < 50 ms. Buffer cho reranker: còn 150 ms. ✅

### Quyết định 6 — Catalog: **Unity Catalog (Databricks) hoặc Apache Polaris**

**Tôi chọn Apache Polaris (self-hosted)** để tránh vendor lock-in.

- **Loại Databricks Unity Catalog** vì: Vendor lock-in — nếu budget giảm, không thể migrate data ra mà không mất fine-grained lineage. Văn phòng luật cần data sovereignty hoàn toàn (tài liệu bảo mật không thể lưu trên Databricks managed storage).
- **Loại Hive Metastore** vì: Không có column-level lineage, không có attribute-based access control (ABAC) — cần ABAC để control "luật sư team A chỉ được đọc hợp đồng dân sự, không được đọc hồ sơ hình sự".
- **Polaris** cung cấp REST Catalog spec → DuckDB, Spark, Trino đều đọc được; hỗ trợ Iceberg + Delta UniForm table registration; fine-grained ABAC per catalog/namespace/table.

---

## 4. Failure Modes — Kịch Bản 3 Giờ Sáng

### FM-1: Embedding Pipeline Crash giữa chừng khi Regenerate v2

**Kịch bản:** Pipeline đang re-embed 1.5B chunks từ v1 → v2. Sau 2 ngày (40% hoàn thành), GPU cluster crash do OOM. Lance dataset `embeddings_v2` đang ở trạng thái incomplete — chứa 600M vectors, thiếu 900M.

**Detection:** 
- Prometheus alert: `embedding_pipeline_last_heartbeat > 5 phút` → PagerDuty.
- Độc lập: Daily reconciliation job so sánh `COUNT(clean_chunks WHERE is_deleted=false)` từ Silver với `COUNT(embeddings_v2)` — lệch > 1% thì alert.

**Rollback (Day 18 concept: Time Travel):**
```
-- Silver: không bị ảnh hưởng (pipeline chỉ đọc Silver, không write)
-- Gold: Lance versioned snapshot
lance_dataset.checkout_version(last_stable_v2_snapshot_id)  # rollback Lance
-- model_registry: không mark v2 là active
-- Query API tiếp tục dùng v1 (không interrupt serving)
```
Lance snapshot cho phép checkout về version ổn định cuối cùng trong < 5 phút. Tiếp tục pipeline từ checkpoint (chunk_id >= last_processed_chunk_id) thay vì restart từ đầu.

---

### FM-2: Tài Liệu Bị Tòa Án Thu Hồi Khẩn Cấp

**Kịch bản:** 3 giờ sáng, Tòa Án Nhân Dân Tối Cao ban hành lệnh thu hồi bản án số 2021/HSST/X. Hệ thống phải xóa toàn bộ chunks của document đó ra khỏi retrieval trong **< 1 tiếng**.

**Detection:** Webhook từ hệ thống quản lý văn bản của Tòa Án (hoặc manual alert từ luật sư trực).

**Rollback (Day 18 concept: Deletion Vectors + ACID):**
```sql
-- Silver: soft-delete với ACID, không rewrite Parquet
UPDATE clean_chunks 
SET is_deleted = true, withdrawn_at = current_timestamp()
WHERE doc_id = '2021/HSST/X';
-- Delta ghi deletion vector — < 1 phút cho 10K chunks

-- Gold: Lance filter-based exclusion (không xóa vật lý ngay)
-- ANN query thêm pre-filter: WHERE is_deleted = false (join với Silver)
-- Hard-delete Lance vectors theo lịch: T+7 ngày (sau khi audit xong)
```
**Audit trail:** Insert vào Delta `audit_withdrawals`: `{doc_id, withdrawn_by, reason, timestamp, chunk_count_affected}`. Immutable (append-only, no UPDATE allowed by governance policy).

---

### FM-3: Schema Evolution — Thêm Multimodal Embedding (CLIP)

**Kịch bản:** Sau 1 năm, team muốn thêm CLIP visual embedding cho image pages. Schema của `embeddings_v3` cần thêm column `clip_vector float32[768]` không có trong v1, v2.

**Detection (preventive):** CI/CD pipeline validate schema compatibility trước khi deploy.

**Rollback plan (Day 18 concept: Schema Evolution):**
```python
# Delta model_registry: schema_mode="merge" để add column an toàn
write_deltalake(
    "model_registry", 
    new_model_row,  # có column mới: clip_dimension=768
    mode="append", 
    schema_mode="merge"  # không break existing readers chỉ đọc text_dimension
)
```
Lance `embeddings_v3` được tạo với schema mới. Query API kiểm tra `model_registry.clip_dimension IS NOT NULL` trước khi thực hiện multimodal search. v1/v2 không bị ảnh hưởng — backward compatible.

---

## 5. Ước Lượng Chi Phí (Back-of-Envelope)

### Storage

| Layer | Data | Size | Tier | $/tháng |
|-------|------|------|------|---------|
| Bronze (Delta) | raw_documents + raw_pages | ~50 TB | S3 Standard-IA | $1,150 |
| Silver (Delta) | clean_chunks + metadata | ~3 TB | S3 Standard | $69 |
| Gold-v1 (Lance, IVF_PQ compressed) | embeddings_v1 | ~2.3 TB | S3 Standard | $53 |
| Gold-v2 (Lance) | embeddings_v2 | ~2.3 TB | S3 Standard | $53 |
| Gold-v3 + future | embeddings_v3 | ~2.5 TB (+ CLIP) | S3 Standard | $58 |
| IVF_PQ index (in-memory server) | centroid table | ~100 GB | EBS GP3 | $8 |
| **Total Storage** | | **~60 TB** | | **~$1,391/tháng** |

*Giả sử: S3 Standard = $0.023/GB; S3 Standard-IA = $0.023/GB first 50 TB. EBS GP3 = $0.08/GB.*

### Compute

| Component | Spec | $/tháng |
|-----------|------|---------|
| Embedding pipeline (GPU, one-time + periodic) | 1× A100 80GB × 7 ngày/lần re-embed × 2 lần/năm | ~$350 amortized |
| Parsing workers (CPU) | 4× c6i.4xlarge (16 vCPU), spot pricing | ~$280 |
| ANN serving (RAM-heavy) | 2× r6i.4xlarge (128 GB RAM) for HA | ~$1,200 |
| LLM API (GPT-4o, ~10K queries/ngày × $0.015/query) | External API | ~$4,500 |
| **Total Compute** | | **~$6,330/tháng** |

**Total: ~$7,721/tháng**  
*(Chủ yếu là LLM API. Nếu self-host Gemma 27B trên A100: LLM cost giảm xuống ~$800/tháng → tổng ~$4,021/tháng)*

---

## 6. MVP Slice — 1 Tuần

**Mục tiêu:** Chứng minh end-to-end pipeline hoạt động trên tập nhỏ 10,000 documents. Không phải production scale — chứng minh kiến trúc work.

**Deliverable sau 1 tuần:**

**Ngày 1–2:** Bronze + Silver pipeline
- Script ingest 10K PDF sample → Bronze Delta table
- pdfplumber + Tesseract OCR → parse text/table/image pages
- Dedup bằng SHA-256 (`MERGE INTO` Delta): đảm bảo idempotent re-run
- Chunking logic: semantic boundary per modality

**Ngày 3–4:** Gold embedding pipeline
- Embed 10K docs × avg 150 chunks/doc = 1.5M chunks với `text-embedding-3-small` (cost: ~$1.5)
- Lưu vào Lance `embeddings_v1` với IVF_PQ index (nlist=256 cho scale nhỏ)
- Lance version snapshot → verify `checkout_version()` khả dụng
- `model_registry` Delta table: register model_id, dimension, created_at

**Ngày 5:** Query + Reproducibility test
- ANN search: top-20 chunks cho 10 legal queries → verify p95 < 200 ms trên 1.5M vectors
- Reproducibility test: ghi lại `{query, embedding_model_id, version_id, chunk_ids}` → 1 ngày sau query lại với pinned version → verify identical results
- Deletion test: withdraw 1 document, verify không xuất hiện trong retrieval sau soft-delete

**Ngày 6–7:** Failure mode drill
- Kill embedding pipeline at 50% → verify Lance rollback → resume from checkpoint
- Schema evolution: add `source_law_article` column vào Silver với `schema_mode="merge"` → verify không break existing queries

**Shippable artifact:** Jupyter notebook end-to-end (có thể chạy từ `make lab` với `pip install lancedb pdfplumber pytesseract`) + kết quả latency benchmark.

**PoC notebook:** Xem `submission/bonus/poc/embedding_version_migration.ipynb`

---

## Self-Checklist (theo rubric)

| Dimension | ✅ Status |
|-----------|---------|
| ≥ 5 quyết định với ≥ 2 alternatives bị loại + tradeoff reasoning | ✅ 6 quyết định (Delta, Lance, Versioning, Chunking, IVF_PQ, Catalog) |
| Scale/latency/budget numbers xuyên suốt document | ✅ 30B tokens, p95 < 200 ms, 1.5B chunks, $7.7K/tháng |
| ≥ 4 Day 18 concepts được áp dụng thực sự | ✅ Medallion, Deletion Vectors + ACID, Time Travel (Lance snapshot), Schema Evolution (`schema_mode="merge"`), Lineage (audit table), FinOps (IVF_PQ compression) |
| ≥ 3 failure modes với detection + rollback cụ thể | ✅ FM1 (pipeline crash), FM2 (withdrawal), FM3 (schema evolution) |
| PoC chạy được, demo phần khó | ✅ `embedding_version_migration.py` demo Lance versioning |
| Failure mode tie với Day 18 concept | ✅ FM1: Time Travel; FM2: Deletion Vectors + ACID; FM3: Schema Evolution |
