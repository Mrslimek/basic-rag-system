## GraphRAG через LlamaIndex (PropertyGraphIndex + Neo4j) — финальный план

### Принцип (записан в память)
Что умеет LlamaIndex нативно — делает LlamaIndex. Самописный графовый стек (`graph_extractor.py`, `graph_neo4j.py`) удаляем, когда нативная ветка заработает.

### Что уже проверено (read-only разведка агента, core 0.14.25)
- `IngestionPipeline` граф-стор НЕ принимает — graph пишет `PropertyGraphIndex.insert_nodes/ainsert` (всего лишь upsert по id чанка; удаление по чанкам — `store.delete_llama_nodes(node_ids=[chunk uuid])`).
- `retriever_mode=`-строк нет: ретривер собирается явно — `index.as_retriever(sub_retrievers=[...])`. Доступные суб-ретриверы: `LLMSynonymRetriever` (скор всегда 1.0) и `VectorContextRetriever` (настоящий cosine). Возвращают обычные `NodeWithScore`.
- `SchemaLLMPathExtractor` дёргает `astructured_predict` → json_schema (тот же путь, что уже работает на llama.cpp) + есть `ImplicitPathExtractor` (без LLM) и текстовый `SimpleLLMPathExtractor` как fallback.
- OpenAILike + `should_use_structured_outputs=True` уже подходит под это без изменений.

### Волна 0-исполнение (агент)
1. `uv add llama-index-graph-stores-neo4j` + `uv pip check`.
2. Читать УСТАНОВЛЕННЫЙ wheel: точная сигнатура `Neo4jPropertyGraphStore` (две конфликтующие версии в природе), labels/rel-типы, требует ли APOC; проверить APOC на живом контейнере и флаг, который его обходит.
3. Живой round-trip проба с именами `zzwave0_test_*` (upsert → get → delete, полная cleanup), существующий Entity/Alias/REL-граф не трогать.
4. Реальный один вызов `SchemaLLMPathExtractor` против нашего llama.cpp 1.5B; при провале — `allow_additional_properties=False`, затем fallback `SimpleLLMPathExtractor`. Результат фиксируется контрактом для волны 1.
Только pyproject/uv.lock + /tmp-скрипты; файлов фич не создаёт.

### Волна 1 — параллельные агенты
- **G1 `app/graph_rag.py` (новый)**: стор (lru_cache), `PropertyGraphIndex.from_existing(store, llm=get_chat_model(), kg_extractors=[SchemaLLMPathExtractor(cхema микросервисов: Service/Endpoint/Database/Queue/Concept; calls/authenticates_via/stores_in/publishes_to...), ImplicitPathExtractor()], embed_model=наш)`, CLI `python -m app.graph_rag sync` (чистка старых граф-узлов этого источника + пересборка из Postgres-чанков, TextNode.id = UUID чанка), `build_graph_retriever(top_k)`.
- **G2 `app/llama_rag.py`**: `aquery(..., retrieve_mode: "vector"|"graph" = "vector")`; графовая ветка: тот же synthesis (RetrieverQueryEngine/response synthesizer + build_llm), `SearchResult` из graph-нод (score: вектор — cosine, graph — 0.0/synonym-1.0 как есть).
- **G3 `main.py` + `frontend.py`**: `retrieve_mode` в `ChatRequest`; в сайдбаре Streamlit radio «RAG mode: Vector / Graph»; skor в UI печатается по режиму.
- **G4 `tests/test_graph_rag.py`**: смоук без LLM: `ImplicitPathExtractor` + засеять чанк-ноды с уникальными именами во временный namespace → `store.delete_llama_nodes` cleanup → retriever вернул чанк-ноды с текстом.
Контракты (имена функций/сигнатуры) пинятся в промптах агентов; один файл = один агент.

### Gate (агент-верификатор, потом я)
Агент: pytest все, живой `graph_rag sync`, `/chat` в обоих режимах на контрольных вопросах (auth/billing — сверяем с прежними 0.88/0.86),Neo4j-граф против Postgres-чанков, cleanup test-мусора. Я читаю diff и коммичу.

### Волна 2 — удаление самописного (агент)
`app/graph_extractor.py`, `app/graph_neo4j.py`, JSONB-колонки `entities/relations`, модели `GraphEntity/EntityAlias/GraphEdge/EdgeMention`, шаг извлечения в worker (переходит на graph_rag-синк), `tests/test_graph_schema.py`, секция Graph в README переписывается. Старый Entity/Alias/REL-граф в Neo4j затирается (DETACH DELETE только его labels) — он был порождением самописного стека.

### Волна 3 — измерение
`eval/questions.md`: table vector vs graph (synonym / synonym+vector) — выбор default-режима за тобой. Модель 1.5B устраивает, в запасе 7B/9B если извлечение окажется тонким.

### Риски
- APOC в контейнере может не быть — волна 0 фиксирует, какой конфигурации стор это обходит.
- wheel neo4j-интеграции на py3.14 — проверка первым шагом, блокер выносится тебе до кода.