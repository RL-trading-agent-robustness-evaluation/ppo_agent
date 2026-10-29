# 首次測試失敗紀錄

首次執行 8 個 unittest 時，2 個通過，6 個 error；共同原因為合成測試的 Builder stub 缺少既有環境需要的 `context_features`（隨後檢查亦補上 `observation_features`）。錯誤為 `AttributeError: 'Builder' object has no attribute 'context_features'`，發生在 parent environment constructor，尚未進入合成交易。

已修正測試 fixture，沒有修改 V11 或改變交易參數。工具回傳首次完整 traceback 已保留於對話；本文件保存原因與處置，後續每次測試輸出保存獨立 log。
